//! Tests for the mapper-ref-texture mapped-surface writers.
//!
//! Out of line for the reason the sibling `runtime/` modules that already do
//! this have: colocated, these 2,291 lines were 48% of `mapping_write.rs` — the
//! module was half test by line count, and the writers themselves were the
//! harder half to find.

use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
use crate::runtime::host::FakeHost;

/// `mapping_geom_window` puts each measurement in the field of its own name.
///
/// `SurfaceWindow`'s four fields are two `u64`s and two `u32`s, so
/// `base_off`/`span_end` can cross silently and so can `bpr`/`bpp`. The
/// mapping below is chosen so all four read differently, which is what
/// makes a crossing observable at all.
///
/// The row pitch is asserted by its relationships, not as a number: a
/// 4-wide BGRA8 surface reports `bpr = 128` against a tight row of 16,
/// because `mapper_ref_texture_sample_window` aligns the pitch up. Hard-coding either
/// value would make this a test of that alignment rather than of which
/// field holds what.
#[test]
fn the_surface_window_names_which_measurement_is_which() {
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mid = 30u32;
    state.map_surface(mid);
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.map_generation = 1;
        m.has_geom = true;
        m.width = 4;
        m.height = 4;
        m.format = MTL_FORMAT_BGRA8_UNORM;
    }
    let rect = Rect {
        origin_x: 0,
        origin_y: 0,
        width: 4,
        height: 4,
    };
    let w = mapping_geom_window(&state, mid, rect).expect("a geometry window");
    assert_eq!(w.bpp, 4, "BGRA8 is four bytes per texel");
    assert!(
        w.bpr >= 4 * w.bpp,
        "a row holds at least the four texels: bpr={} bpp={}",
        w.bpr,
        w.bpp
    );
    assert_ne!(
        w.bpr, w.bpp,
        "the two must differ here or this test could not see them swapped"
    );
    assert_eq!(
        w.span_end - w.base_off,
        u64::from(w.bpr) * 4,
        "the span reaches exactly the four rows the rectangle asked for"
    );
    // A rectangle past the declared extent has no window at all.
    assert!(mapping_geom_window(
        &state,
        mid,
        Rect {
            origin_x: 1,
            ..rect
        }
    )
    .is_none());
}

/// The gap walk is what both writeback paths subtract their skipped ranges
/// with, so it is tested on its own: an off-by-one here is a row of pixels
/// written into a page the guest owns, or a row of the guest's own bytes
/// overwritten, and neither is visible in a frame until much later.
#[test]
fn unskipped_returns_exactly_the_bytes_no_range_covers() {
    assert_eq!(unskipped(0, 10, &[]), vec![(0, 10)]);
    assert_eq!(unskipped(10, 10, &[]), vec![]);
    // Interior, leading, trailing, and total cover.
    assert_eq!(unskipped(0, 10, &[(4, 6)]), vec![(0, 4), (6, 10)]);
    assert_eq!(unskipped(0, 10, &[(0, 4)]), vec![(4, 10)]);
    assert_eq!(unskipped(0, 10, &[(6, 10)]), vec![(0, 6)]);
    assert_eq!(unskipped(0, 10, &[(0, 10)]), vec![]);
    // Ranges outside the window contribute nothing, on either side.
    assert_eq!(unskipped(10, 20, &[(0, 5), (30, 40)]), vec![(10, 20)]);
    // Partial overlap at both ends, and several ranges in one window.
    assert_eq!(
        unskipped(10, 30, &[(5, 12), (16, 18), (28, 35)]),
        vec![(12, 16), (18, 28)]
    );
}

/// A rect taller than the window it names is refused on **both** storage
/// shapes, and writes nothing past the window.
///
/// `write_rect_raw_at_impl` has three arms that all write guest memory, and
/// the bound used to be on two of them. The per-row fragmented arm reached
/// `mapper::write_mapping_bytes`, which bounds against the whole mapping's
/// page span rather than this plane's window, so an over-tall rect landed in
/// whatever follows the window — on a multi-plane IOSurface, the next plane's
/// pixels — with no fail line. The packed arm refused the same call.
///
/// So the loop over `packed` is the test: an assertion on one shape alone
/// passed throughout, which is how the hole survived. `span_end` is set short
/// of the mapping's real extent on purpose, because that gap is exactly the
/// region the unbounded arm wrote into and the bounded one did not.
#[test]
fn a_rect_taller_than_its_window_is_refused_on_both_storage_shapes() {
    use crate::model::PAGE_SHIFT_X86;
    const PAGE: u64 = 1 << PAGE_SHIFT_X86;
    const W: u32 = 16;
    const BPR: u32 = W * 4;

    for packed in [true, false] {
        let mut state = DeviceState::new(DeviceId(3), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = !packed;
        let base_pfn = 0x60u32;
        host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 4 * PAGE as usize, 0xee);
        let order: Vec<u32> = if packed {
            (0..4).collect()
        } else {
            vec![3, 2, 1, 0]
        };
        let entries: Vec<u32> = order
            .iter()
            .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
            .collect();
        state.map_surface(5);
        state.attach_mapping_internal(5, 0);
        let m = state.mappings.get_mut(&5).unwrap();
        m.mapping_internal = 1;
        m.page_entries = entries;

        // The window is one page: 64 rows of 64 bytes. The rect asks for 80.
        let span_end = PAGE;
        let rows = (PAGE / BPR as u64) as u32;
        let over = rows + 16;
        let src = vec![0x5au8; (over as usize) * BPR as usize];

        assert!(
            !write_rect_raw_at(
                &mut state,
                &mut host,
                5,
                SurfaceWindow {
                    base_off: 0,
                    bpr: BPR,
                    span_end,
                    bpp: 4
                },
                Rect {
                    origin_x: 0,
                    origin_y: 0,
                    width: W,
                    height: over
                },
                &src,
                BPR
            ),
            "packed={packed}: a rect past the window's last row must be refused"
        );

        // The bytes after the window still hold the fill the mapping was
        // seeded with, on both shapes.
        let mut after = [0u8; 16];
        assert!(mapper::read_mapping_bytes(
            &mut state, &mut host, 5, span_end, &mut after
        ));
        assert_eq!(
            after, [0xeeu8; 16],
            "packed={packed}: the refused rect must not have written past the window"
        );
    }
}

/// The skip test below fills the frame with one repeated byte, so it proves
/// which *pages* were written and nothing at all about which row went
/// where: a writeback that repeated row 0 sixty-four times, or that shifted
/// every row by one, passes it unchanged. That is not hypothetical — a
/// BGRA8 row no longer passes through the conversion scratch buffer on its
/// way to the guest, so the source offset, the tight row length and the
/// destination stride are now read from three places that used to be two.
///
/// Both storage shapes, because they place rows by different means, and a
/// non-BGRA format so the staged path is exercised beside the direct one.
#[test]
fn a_writeback_lands_every_row_at_its_own_offset() {
    use crate::model::PAGE_SHIFT_X86;
    const PAGE: u64 = 1 << PAGE_SHIFT_X86;
    const W: u32 = 64;
    const H: u32 = 64;
    const BPR: usize = (W * 4) as usize;

    for packed in [true, false] {
        for format in [MTL_FORMAT_BGRA8_UNORM, pixel_format::MTL_FORMAT_RGBA8_UNORM] {
            let mut state = DeviceState::new(DeviceId(2), PAGE_SHIFT_X86);
            let mut host = FakeHost::new();
            host.strict_linux_map = !packed;
            let base_pfn = 0x40u32;
            host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 8 * PAGE as usize, 0);
            let order: Vec<u32> = if packed {
                (0..4).collect()
            } else {
                vec![3, 2, 1, 0]
            };
            let entries: Vec<u32> = order
                .iter()
                .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
                .collect();
            state.map_surface(4);
            state.attach_mapping_internal(4, 0);
            let m = state.mappings.get_mut(&4).unwrap();
            m.mapping_internal = 1;
            m.page_entries = entries;
            assert!(state.set_mapping_geom(4, W, H, format));

            // Row y is filled with the byte y in every channel, so a row
            // that lands at the wrong offset is visible as the wrong value
            // and a duplicated row is visible as a repeat. Channel order
            // does not matter to that, which is what lets one frame test
            // both formats.
            let mut frame = vec![0u8; (W * H * 4) as usize];
            for y in 0..H as usize {
                frame[y * BPR..(y + 1) * BPR].fill(y as u8);
            }
            assert!(write_bgra8_skipping(
                &mut state,
                &mut host,
                4,
                &frame,
                W * 4,
                W,
                H,
                &[]
            ));

            for y in 0..H as usize {
                // Which guest page this row's first byte lives in, and where
                // in it, walking the same page list the mapping declares.
                let off = y * BPR;
                let gpa = ((base_pfn as u64 + order[off / PAGE as usize] as u64) << PAGE_SHIFT_X86)
                    + (off as u64 % PAGE);
                let mut got = [0u8; 4];
                host.read_gpa(gpa, &mut got).unwrap();
                assert!(
                    got.iter().all(|b| *b == y as u8),
                    "packed={packed} fmt={format:#x} row {y} must read {y:#x}, got {got:?}"
                );
            }
        }
    }
}

/// A writeback told to skip a page leaves that page exactly as the guest
/// left it, writes every other page, and stops claiming the frame is the
/// mapping's content.
///
/// Both storage shapes are exercised, because they place bytes by different
/// means — the packed one pokes a host view through a raw pointer, the
/// fragmented one stages a frame and hands runs to the mapper — and a skip
/// honoured by one and not the other is a defect that only appears on
/// whichever guest allocation happens to be scattered.
///
/// Dropping the host cache at the tail is half the test: it would otherwise
/// answer the mapper-ref-texture LOAD seed with the very bytes the guest's stores were
/// preserved from.
#[test]
fn a_skipping_writeback_leaves_the_guest_its_own_pages() {
    use crate::model::PAGE_SHIFT_X86;
    const PAGE: u64 = 1 << PAGE_SHIFT_X86;
    // 64x64 BGRA8 is 256 bytes a row, 16 KiB in four x86 pages.
    const W: u32 = 64;
    const H: u32 = 64;

    for packed in [true, false] {
        let mut state = DeviceState::new(DeviceId(2), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = !packed;
        let base_pfn = 0x40u32;
        host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 8 * PAGE as usize, 0x55);
        // Reversed page order is a fragmented list to `map_pages`; the same
        // four pages either way, so the guest bytes are comparable.
        let order: Vec<u32> = if packed {
            (0..4).collect()
        } else {
            vec![3, 2, 1, 0]
        };
        let entries: Vec<u32> = order
            .iter()
            .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
            .collect();
        state.map_surface(4);
        state.attach_mapping_internal(4, 0);
        let m = state.mappings.get_mut(&4).unwrap();
        m.mapping_internal = 1;
        m.page_entries = entries;
        assert!(state.set_mapping_geom(4, W, H, MTL_FORMAT_BGRA8_UNORM));
        crate::runtime::surface_cache::store(&mut state, 4, W, H, vec![0u8; (W * H * 4) as usize]);

        let frame = vec![0xAAu8; (W * H * 4) as usize];
        // Page 2 of the surface, in mapping-offset space.
        let skip = [(2 * PAGE, 3 * PAGE)];
        assert!(write_bgra8_skipping(
            &mut state,
            &mut host,
            4,
            &frame,
            W * 4,
            W,
            H,
            &skip
        ));

        for page in 0..4u64 {
            let gpa = (base_pfn as u64 + order[page as usize] as u64) << PAGE_SHIFT_X86;
            let mut got = [0u8; 8];
            host.read_gpa(gpa, &mut got).unwrap();
            let want = if page == 2 { 0x55 } else { 0xAA };
            assert!(
                got.iter().all(|b| *b == want),
                "packed={packed} surface page {page} must read {want:#x}, got {got:?}"
            );
        }
        assert!(
            crate::runtime::surface_cache::get(&state, 4, W, H).is_none(),
            "packed={packed}: the cache must stop answering for a mapping the frame no longer describes"
        );
    }
}

/// A host write into guest RAM must announce itself, because the hypervisor's
/// dirty bitmap is defined not to see it.
///
/// The bitmap witnesses guest CPU stores. Everything this device puts into
/// the same pages is invisible to it, so a reader holding "the guest has not
/// written since I looked" would keep serving a copy that *we* superseded.
/// `DeviceState::host_writes` is the only thing that separates the two, and a
/// writer that forgets to record there is silent in exactly the way that
/// matters.
///
/// The read half is the other half of the contract: reads share the same
/// mapping walk, and a read that moved the record would make every reader
/// re-fetch on account of a reader.
/// A dropped writeback must say which check dropped it.
///
/// The composite surface is the largest frame this device moves, and losing
/// one is a wrong picture that then persists. Sixteen refusal sites used to
/// answer with a bare `false` that the caller rendered as one
/// `reason=write_refused`, so a reader could tell that the frame had been
/// dropped and nothing else.
///
/// `GeometryMoved` is the one this test drives because it is the one a
/// deferred window can reach without anything being broken: the frame is
/// armed at one rect and the surface is re-published at another — which is
/// what an appearance change or a wallpaper switch does. Asserting the
/// *specific* route is the whole point; asserting `!ok` would pass with every
/// site sharing a slug again.
#[test]
fn a_writeback_refused_because_the_geometry_moved_says_so_by_name() {
    use crate::model::PAGE_SHIFT_X86;
    const PAGE: u64 = 1 << PAGE_SHIFT_X86;

    let mut state = DeviceState::new(DeviceId(3), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let base_pfn = 0x40u32;
    host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 8 * PAGE as usize, 0x55);
    state.map_surface(4);
    state.attach_mapping_internal(4, 0);
    let m = state.mappings.get_mut(&4).unwrap();
    m.mapping_internal = 1;
    m.page_entries = (0..4)
        .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
        .collect();
    // Latched at 64x64; the frame below is armed at 32x32, as a window armed
    // before a re-publish would be.
    assert!(state.set_mapping_geom(4, 64, 64, MTL_FORMAT_BGRA8_UNORM));

    // Deltas, not absolutes: the route counters are process-global and every
    // other test in this binary shares them, so a sibling reading nonzero
    // says nothing about this write.
    const SIBLINGS: [&str; 3] = [
        "surface_write_mapping_absent",
        "surface_write_pages_not_ours",
        "surface_write_source_short",
    ];
    let before = crate::runtime::drain::store_route_count("surface_write_geometry_moved");
    let before_siblings: Vec<u64> = SIBLINGS
        .iter()
        .map(|r| crate::runtime::drain::store_route_count(r))
        .collect();

    let frame = vec![0xAAu8; 32 * 32 * 4];
    assert!(
        !write_bgra8(&mut state, &mut host, 4, &frame, 32 * 4, 32, 32),
        "a frame whose rect is not the surface's must not land"
    );
    assert_eq!(
        crate::runtime::drain::store_route_count("surface_write_geometry_moved"),
        before + 1,
        "the writeback was dropped without naming the check that dropped it"
    );
    // And no sibling check moved, or the slug is not discriminating.
    for (route, was) in SIBLINGS.iter().zip(before_siblings) {
        assert_eq!(
            crate::runtime::drain::store_route_count(route),
            was,
            "{route} fired for a geometry mismatch"
        );
    }
}

#[test]
fn writing_guest_pages_moves_the_host_write_record_and_reading_them_does_not() {
    use crate::model::PAGE_SHIFT_X86;
    const PAGE: u64 = 1 << PAGE_SHIFT_X86;
    const W: u32 = 64;
    const H: u32 = 64;

    let mut state = DeviceState::new(DeviceId(3), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let base_pfn = 0x40u32;
    host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 8 * PAGE as usize, 0x55);
    let entries: Vec<u32> = (0..4)
        .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
        .collect();
    state.map_surface(4);
    state.attach_mapping_internal(4, 0);
    let m = state.mappings.get_mut(&4).unwrap();
    m.mapping_internal = 1;
    m.page_entries = entries;
    assert!(state.set_mapping_geom(4, W, H, MTL_FORMAT_BGRA8_UNORM));

    let before = state.host_writes.epoch();
    let frame = vec![0xAAu8; (W * H * 4) as usize];
    assert!(write_bgra8(&mut state, &mut host, 4, &frame, W * 4, W, H));
    assert_ne!(
        state.host_writes.epoch(),
        before,
        "a write into the guest's pages went unannounced"
    );

    let after_write = state.host_writes.epoch();
    let mut out = vec![0u8; (W * H * 4) as usize];
    assert!(crate::runtime::mapper::read_mapping_bytes(
        &mut state, &mut host, 4, 0, &mut out
    ));
    assert_eq!(
        state.host_writes.epoch(),
        after_write,
        "a read moved the write record, so every reader now invalidates every reader"
    );
}

/// A skipping write must re-take the guest-write stamp, or the set of pages
/// it skips grows monotonically until it covers the whole surface and the
/// device's composite never reaches guest memory again.
///
/// `since` for every later `guest_written_pages` call is this stamp, and the
/// host's `page_gen` records the harvest that last saw each page written and
/// never resets per consumer. So a stamp that does not move keeps naming
/// every page the guest has *ever* written since it was taken. One full CPU
/// repaint of a window marks all of them, and from then on each deferred
/// flush of that surface skips the entire extent.
///
/// That is what this rail shipped with, and it is a desktop that goes black
/// and stays black — reproduced live at `render_flush_preserved_guest_write`
/// ~65 a second on a boot with the sampled resident rung gated off, which is
/// what placed it here rather than there.
///
/// The stamp is honest because of what runs beside it: the byte cache is
/// dropped and the resident's content claim withdrawn, so when it says "no
/// host-side copy is known stale relative to these pages" there is no
/// host-side copy at all.
/// A write that must not touch some ranges may not discharge the mapping's
/// writeback debt by *paying* it, because the payment has no skip list.
///
/// `merge_guest_writes_into_pages` is the ladder's whole correction for "the
/// guest CPU painted this surface under a live resident": read the resident,
/// land it in every page the guest did *not* write, keep the pages it did. It
/// gets there through `write_bgra8_skipping`, whose first act was
/// `settle_for_mapping` — and that pays the owed frame over the **whole** window
/// with no exclusions. So the guest's pages were overwritten one statement before
/// the skipping write restored everything around them, the merge reported
/// success, and the repaint the merge exists to preserve was gone.
///
/// The debt is superseded instead: the caller is about to land the same
/// surface's newer content at the same geometry, which `GeometryMoved` already
/// enforces, so the owed frame is replaced rather than lost.
#[test]
fn a_skipping_write_supersedes_the_debt_instead_of_paying_it_over_the_skip() {
    use crate::model::PAGE_SHIFT_X86;
    const PAGE: u64 = 1 << PAGE_SHIFT_X86;
    const W: u32 = 64;
    const H: u32 = 64;

    let mut state = DeviceState::new(DeviceId(7), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let base_pfn = 0x90u32;
    host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 8 * PAGE as usize, 0x55);
    let entries: Vec<u32> = (0..4)
        .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
        .collect();
    state.map_surface(7);
    state.attach_mapping_internal(7, 0);
    let m = state.mappings.get_mut(&7).unwrap();
    m.mapping_internal = 1;
    m.page_entries = entries;
    assert!(state.set_mapping_geom(7, W, H, MTL_FORMAT_BGRA8_UNORM));
    let map_generation = state.mappings[&7].map_generation;

    let count = |r: &str| crate::runtime::drain::store_route_count(r);
    let paid0 = count("wbdebt_paid_named");
    let superseded0 = count("wbdebt_superseded_by_skipping_write");

    // A Store owes this mapping a frame.
    assert!(
        state
            .pending_writebacks
            .arm(
                7,
                crate::runtime::writeback_debt::test_resident_identity(
                    7,
                    W,
                    H,
                    u64::from(map_generation),
                ),
                W,
                H,
                map_generation,
            )
            .is_none(),
        "one debt cannot overflow a ledger of 32"
    );

    let frame = vec![0xAAu8; (W * H * 4) as usize];
    assert!(write_bgra8_skipping(
        &mut state,
        &mut host,
        7,
        &frame,
        W * 4,
        W,
        H,
        &[(2 * PAGE, 3 * PAGE)]
    ));

    assert!(
        state.pending_writebacks.get(7).is_none(),
        "the debt must not survive: the frame it owes has just been landed"
    );
    assert_eq!(
        count("wbdebt_superseded_by_skipping_write"),
        superseded0 + 1,
        "the debt is superseded by the write that is about to land the same \
         surface, and the route is what prices how often the two co-occur"
    );
    assert_eq!(
        count("wbdebt_paid_named"),
        paid0,
        "paying it writes the owed frame over the skip list, which is exactly \
         the guest repaint this write was called to preserve"
    );

    // The whole-frame writer is unchanged and still pays: it has no ranges to
    // protect, and a debt left standing there would be read straight past.
    //
    // Paying is the Vulkan rail: `writeback_debt::pay` lands the owed frame out
    // of a resident the engine holds, and the arm with no engine has no such
    // frame — its `pay` is the unreachable stub, because nothing on that arm
    // arms the ledger in the first place. The supersede half above is the
    // arm-independent claim and stays ungated; this half asserts a payment only
    // the engine can make, so it is checked where the payer exists.
    #[cfg(feature = "backend-vulkan")]
    {
        assert!(state
            .pending_writebacks
            .arm(
                7,
                crate::runtime::writeback_debt::test_resident_identity(
                    7,
                    W,
                    H,
                    u64::from(map_generation),
                ),
                W,
                H,
                map_generation,
            )
            .is_none());
        assert!(write_bgra8(&mut state, &mut host, 7, &frame, W * 4, W, H));
        assert_eq!(
            count("wbdebt_paid_named"),
            paid0 + 1,
            "a write with nothing to skip must still discharge the debt by paying it"
        );
    }
}

#[test]
fn a_skipping_writeback_re_takes_the_stamp_so_the_skip_set_cannot_only_grow() {
    use crate::model::PAGE_SHIFT_X86;
    const PAGE: u64 = 1 << PAGE_SHIFT_X86;
    const W: u32 = 64;
    const H: u32 = 64;

    let mut state = DeviceState::new(DeviceId(3), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let base_pfn = 0x40u32;
    host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 8 * PAGE as usize, 0x55);
    let entries: Vec<u32> = (0..4)
        .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
        .collect();
    state.map_surface(4);
    state.attach_mapping_internal(4, 0);
    let m = state.mappings.get_mut(&4).unwrap();
    m.mapping_internal = 1;
    m.page_entries = entries;
    assert!(state.set_mapping_geom(4, W, H, MTL_FORMAT_BGRA8_UNORM));

    // A Store arms the witness, then the guest paints page 2 of the surface.
    crate::runtime::mapper::stamp_guest_write_gen(&mut state, &mut host, 4);
    let token = state.mappings[&4].guest_write_token;
    assert_ne!(token, 0, "the fake host must be able to watch these pages");
    host.guest_wrote_page((base_pfn as u64 + 2) << PAGE_SHIFT_X86);

    let stamped_before = state.mappings[&4].guest_write_gen_at_store;
    assert!(
        !host
            .guest_written_pages(token, stamped_before)
            .unwrap()
            .is_empty(),
        "the guest's write must be visible against the stamp the flush would use"
    );

    let frame = vec![0xAAu8; (W * H * 4) as usize];
    assert!(write_bgra8_skipping(
        &mut state,
        &mut host,
        4,
        &frame,
        W * 4,
        W,
        H,
        &[(2 * PAGE, 3 * PAGE)]
    ));

    let stamped_after = state.mappings[&4].guest_write_gen_at_store;
    assert_ne!(
        stamped_after, 0,
        "a skipping write that leaves the stamp at 0 makes every later flush skip everything"
    );
    assert_eq!(
        host.guest_written_pages(token, stamped_after),
        Some(Vec::new()),
        "the write the device just honoured must not be named again by the next flush"
    );
}

#[test]
fn write_bumps_generation() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x10u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(3);
    state.attach_mapping_internal(3, 0); // leave internal 0; set pages manually
    let m = state.mappings.get_mut(&3).unwrap();
    m.mapping_internal = 1;
    m.page_entries = vec![entry];
    assert!(state.set_mapping_geom(3, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    let src = [0x11u8, 0x22, 0x33, 0x44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    // 2x2 BGRA, stride 8
    assert!(write_bgra8(&mut state, &mut host, 3, &src, 8, 2, 2));
    assert_eq!(state.mappings.get(&3).unwrap().content_generation, 1);
}

/// The write that makes the host copy authoritative must also arm the
/// witness for it.
///
/// This function writes the guest pages and then stores the host render
/// cache, so at this instant the two agree — the one moment the copy's
/// currency can be pinned. Nothing else armed it: the backing sampled
/// ladder's first census read `t11rung_host_cache_gw_no_stamp` 14 092
/// against `gw_clean` 0, because only the Vulkan Store rails ever stamped
/// while the copy that rung serves is written here. Unstamped, the reader
/// cannot tell a surface the guest has rewritten from one it has not, and
/// has to assume the worst on every bind.
#[test]
fn a_host_cache_write_arms_the_guest_write_witness_for_the_copy() {
    use crate::runtime::host::HostOps;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x30u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(11);
    state.attach_mapping_internal(11, 0);
    let m = state.mappings.get_mut(&11).unwrap();
    m.mapping_internal = 1;
    m.page_entries = vec![entry];
    assert!(state.set_mapping_geom(11, 2, 2, MTL_FORMAT_BGRA8_UNORM));

    assert_eq!(
        state.mappings[&11].guest_write_token, 0,
        "nothing has armed this mapping yet"
    );
    let src = [0x11u8, 0x22, 0x33, 0x44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    assert!(write_bgra8(&mut state, &mut host, 11, &src, 8, 2, 2));

    let token = state.mappings[&11].guest_write_token;
    assert_ne!(token, 0, "the store must register the pages it copied");
    assert_eq!(
        host.guest_write_gen(token),
        Some(state.mappings[&11].guest_write_gen_at_store),
        "the recorded generation must be the one the copy is current as of"
    );

    // A guest CPU store into the surface, with no device operation: the
    // recorded generation no longer matches, which is exactly what the
    // sampled ladder reads to refuse the copy.
    host.guest_wrote_page(gpa);
    assert_ne!(
        host.guest_write_gen(token),
        Some(state.mappings[&11].guest_write_gen_at_store),
        "a guest write must move the host's generation away from the stamp"
    );
}

/// A guest write drops only the storage-residency mirror windows it
/// intersects; disjoint sibling windows (ping-pong canvases) survive.
#[test]
fn mapping_write_invalidates_intersecting_residency_windows_only() {
    use crate::model::ComputeStorageResidencyKey;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(7);
    state.attach_mapping_internal(7, 0);
    let m = state.mappings.get_mut(&7).unwrap();
    m.mapping_internal = 1;
    m.page_entries = vec![entry];
    let window = |surface_offset: u64, span_end: u64| ComputeStorageResidencyKey {
        mapping_id: 7,
        map_generation: state.mappings[&7].map_generation,
        surface_offset,
        surface_bpr: 32,
        span_end,
        width: 8,
        height: 2,
        pixel_format: MTL_FORMAT_BGRA8_UNORM,
        texture_ref: 0,
    };
    let hit = window(0, 64);
    let survivor = window(1024, 1088);
    state.compute_storage_residency.insert(hit, 5);
    state.compute_storage_residency.insert(survivor, 5);
    let vouched = mapper::vouch_mapping_pages_verdict(&mut state, &host, 7)
        .1
        .expect("no walk to contradict");
    assert!(mapper::write_mapping_bytes(
        &mut state, &mut host, 7, 16, &[0u8; 32], &vouched
    ));
    assert!(!state.compute_storage_residency.contains_key(&hit));
    assert!(state.compute_storage_residency.contains_key(&survivor));
}

/// A direct mapper-ref-texture writeback must not land in a page the guest re-pointed
/// away, and this asserts it in the currency of the bug: the bytes of the
/// page the surface moved to.
///
/// The page-drift witness shipped with exactly one caller — the deferred
/// render flush — so this rail, which writes a full frame of pixels through
/// `MappingEntry::page_entries`, was unguarded. The crash reports are the
/// receipt: WindowServer aborting in `small_free_list_remove_ptr_no_clear`,
/// and guest-kernel kalloc poison finding whole freed elements filled with
/// `0xff` from offset 0 — opaque white BGRA in memory already handed to
/// somebody else.
///
/// So the fixture recycles a page the way the guest does: adopt a list walked
/// through a live task page table, write a frame (which must land), re-point
/// the PTE with no packet — nothing bumps `map_generation`, which is the
/// whole defect — and require the second write to refuse. `data1` stands for
/// whatever the guest gave those pages to next, seeded with a pattern rather
/// than zeroes so "refused" and "wrote zeroes" cannot be confused.
#[test]
fn a_repointed_surface_refuses_the_write_and_leaves_the_new_owner_alone() {
    use crate::model::{BackingWalk, PAGE_SHIFT_X86};
    use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let page = 1u64 << PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data0 = 4u64 << PAGE_SHIFT_X86;
    let data1 = 10u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, page as usize, 0);
    host.map_range(root_gpa, page as usize, 0);
    host.map_range(data0, page as usize, 0);
    host.map_range(data1, page as usize, 0);

    let st32 = |b: &mut [u8], v: u32| b[..4].copy_from_slice(&v.to_le_bytes());
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(dir_gpa, &d).unwrap();
    let mut pte = [0u8; 4];
    st32(&mut pte, (data0 >> PAGE_SHIFT_X86) as u32);
    host.write_gpa(root_gpa, &pte).unwrap();

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state.define_task(1, page, 2);
    let mid = 6;
    state.map_surface(mid);
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.mapped = true;
        m.page_entries =
            vec![(((data0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        m.backing_walk = Some(BackingWalk {
            task_id: 1,
            backing_pfn: 0,
            map_generation: m.map_generation,
        });
    }

    // A tight 8x4 BGRA8 frame of opaque white — the payload the crash census
    // reads back out of freed guest memory.
    let (w, h) = (8u32, 4u32);
    let frame = vec![0xffu8; (w * h * RGBA8_BPP) as usize];
    let stride = w * RGBA8_BPP;
    assert!(
        write_bgra8(&mut state, &mut host, mid, &frame, stride, w, h),
        "the list was just walked from this page table, so the frame lands"
    );
    let mut landed = [0u8; 16];
    host.read_gpa(data0, &mut landed).unwrap();
    assert_eq!(landed, [0xffu8; 16], "the first frame reached data0");

    // Now the guest reclaims that page and hands it to something else, which
    // writes its own bytes there — a malloc small-zone region, say, whose
    // free-list pointers live *inside* the freed blocks. `0x5a` stands for
    // them. This is the step the crash reports are about: the corruption
    // lands in the page the surface *left*, not the one it moved to, because
    // the device's cached contig view is a `mach_vm_remap` of the old PFNs
    // and keeps resolving there.
    host.write_gpa(data0, &[0x5au8; 16]).unwrap();

    // And the surface is re-pointed. No MapMemory2, no UnmapMemory, no
    // ReplacePhysical — nothing on the wire, so nothing bumps the
    // incarnation.
    let generation_before = state.mappings.get(&mid).unwrap().map_generation;
    st32(&mut pte, (data1 >> PAGE_SHIFT_X86) as u32);
    host.write_gpa(root_gpa, &pte).unwrap();
    assert_eq!(
        state.mappings.get(&mid).unwrap().map_generation,
        generation_before,
        "no packet arrived, so nothing bumped the incarnation"
    );

    let refused = !write_bgra8(&mut state, &mut host, mid, &frame, stride, w, h);
    // The memory assertion comes first deliberately. A return value is this
    // crate's opinion about what it did; `data0` is what the guest will
    // actually find in its heap, and that is the claim the crash reports
    // dispute. Asserting the opinion first would let a rail that returns
    // false *after* writing pass the stronger half unread.
    let mut recycled = [0u8; 16];
    host.read_gpa(data0, &mut recycled).unwrap();
    assert_eq!(
        recycled, [0x5au8; 16],
        "the page the guest took away must still hold its new owner's bytes \
         — this is the guest heap corruption the whole goal is about"
    );
    assert!(
        refused,
        "and the caller is told, so a lost frame is never read as a landed one"
    );

    // Refusing once is not enough: `page_entries` is what every later reader
    // and writer resolves through, so a list a fresh walk has contradicted
    // has to stop being believed rather than be skipped once.
    assert!(
        state.mappings.get(&mid).unwrap().page_entries.is_empty(),
        "the contradicted list is dropped, so the next bind re-resolves"
    );
    assert_ne!(
        state.mappings.get(&mid).unwrap().map_generation,
        generation_before,
        "and every window still armed against the old incarnation refuses \
         on the map_generation check it already had"
    );
}

/// compute_writeback_amplification: fragmented texture writeback imports
/// once per maximal GPA run, not once per image row.
#[test]
fn fragmented_raw_rect_bulk_imports_runs_not_rows() {
    use crate::model::PAGE_SHIFT_X86;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1usize << PAGE_SHIFT_X86;
    let gpa0 = 0x1000_0000u64;
    let gpa1 = 0x2000_0000u64;
    host.map_range(gpa0, page, 0x7e);
    host.map_range(gpa1, page, 0x7e);
    let mid = 19;
    state.map_surface(mid);
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.page_entries = vec![
            (((gpa0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            (((gpa1 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    let src = vec![0x2a; 4 * 16];
    // Contract span ends at the last texel, excluding trailing row
    // padding: (height - 1) * bpr + tight = 3 * 2048 + 16.
    assert!(write_full_rect_raw_at(
        &mut state, &mut host, mid, 0, 2048, 6160, 4, 4, 4, &src, 16,
    ));
    // One full-view attempt plus one successful import per maximal GPA run.
    // The refusal is cached for this page-list generation, so the extra call
    // is constant rather than per row. The old row loop took nine attempts for
    // these four rows and scaled with height.
    assert_eq!(host.map_pages_calls, 3);
    let calls_after_write = host.map_pages_calls;

    let mut row = [0u8; 16];
    assert!(mapper::read_mapping_bytes(
        &mut state, &mut host, mid, 4096, &mut row,
    ));
    assert_eq!(row, [0x2a; 16]);
    assert_eq!(calls_after_write, 3);
}

/// The BGRA row writers reach `observe::footprint`.
///
/// This is the rail the first cut of the footprint's completeness gate
/// missed, and it is the biggest one in the device. These writers never call
/// `mapper::write_mapping_bytes` and never call `HostOps::map_pages`: they
/// take a contig view through `contig_for_write` and poke rows straight into
/// it. A gate that scanned for `map_pages` callers therefore scored this
/// file as reaching guest RAM by no mechanism at all — which was true of the
/// needle and false of the file.
///
/// A missing mark here would have left the footprint empty of nearly every
/// pixel this device writes, and an empty set answers "we never wrote there"
/// to every panic it is asked about.
#[test]
fn a_bgra_row_write_marks_the_footprint_through_its_contig_view() {
    use crate::model::PAGE_SHIFT_X86;
    use crate::observe::footprint;

    let _fp = footprint::exclusive_for_tests();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page = 1u64 << PAGE_SHIFT_X86;
    // Adjacent so the contig view packs — this is the path production takes.
    let gpa0 = 0x5000_0000u64;
    host.map_range(gpa0, 2 * page as usize, 0);
    let mid = 12u32;
    state.map_surface(mid);
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![
            (((gpa0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            ((((gpa0 + page) >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    assert!(
        mapper::ensure_contig_view(&mut state, &mut host, mid).is_some(),
        "the fixture must take the contig path or it tests the other rail"
    );
    let src = [0xFFu8; 16];
    assert!(write_bgra8(&mut state, &mut host, mid, &src, 8, 2, 2));

    assert!(
        footprint::wrote_gpa(gpa0),
        "the surface's first frame must be in the set"
    );
    assert!(
        !footprint::wrote_gpa(gpa0 + 8 * page),
        "and nothing beyond the surface"
    );
}

/// Linux product: non-packed page list still lands BGRA via multi-import.
#[test]
fn write_bgra8_fragmented_pages_multi_import() {
    use crate::model::PAGE_SHIFT_X86;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    // 2×2 BGRA needs 16 bytes → one page; use two non-adjacent pages so
    // ensure_contig_view fails and multi-import is forced.
    let gpa0 = 0x3000_0000u64;
    let gpa1 = 0x4000_0000u64;
    host.map_range(gpa0, page as usize, 0);
    host.map_range(gpa1, page as usize, 0);
    let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
    let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
    let mid = 11u32;
    state.map_surface(mid);
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![
            (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    let src = [
        0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        0x10,
    ];
    assert!(write_bgra8(&mut state, &mut host, mid, &src, 8, 2, 2));
    let mut first = [0u8; 4];
    assert!(host.read_gpa(gpa0, &mut first).is_ok());
    assert_eq!(&first, &src[..4]);
    assert!(mapper::ensure_contig_view(&mut state, &mut host, mid).is_none());
}

/// The fragmented BGRA write path must land the texels and nothing else —
/// the same contract `write_bgra8_contig_writes_only_inside_the_sample_window`
/// pins on the pointer arm, asserted the same way so the two arms cannot
/// drift apart. Padding after the final row would overrun an exact IOSurface
/// allocation; padding *between* rows is inside the plane but is still
/// content the guest put there and this call never named.
/// The staged full-plane rect write must leave inter-row padding alone, on
/// the same assertion as its two siblings.
///
/// `write_rect_raw_at_impl` has three arms that must land identical guest
/// memory. The contiguous one pokes each row's texel bytes through a raw
/// pointer, the per-row fragmented one writes each row's bytes on its own,
/// and the staged one built a pitch-wide zeroed frame and stored it entire —
/// so every padding byte between rows was zeroed in the guest's pages. That
/// is the defect `write_bgra8_inner` was fixed for, in the sibling function,
/// and this arm's own comment cited the pre-fix behaviour as its
/// justification.
///
/// Asserted as "every byte outside the texel runs is unchanged", the same
/// whole-page comparison the BGRA arms are pinned by, so all three cannot
/// drift apart again. The fixture forces the staged arm two ways: two
/// non-adjacent pages (no contiguous view) and `full_plane`, and a pitch
/// wider than the packed row so `full_tight_direct` cannot take it either.
#[test]
fn write_full_rect_raw_staged_leaves_inter_row_padding_alone() {
    use crate::model::PAGE_SHIFT_X86;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    let gpa0 = 0x3500_0000u64;
    let gpa1 = 0x4600_0000u64;
    host.map_range(gpa0, page as usize, 0xCC);
    host.map_range(gpa1, page as usize, 0xCC);
    let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
    let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
    let mid = 14u32;
    state.map_surface(mid);
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![
            (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    assert!(
        mapper::ensure_contig_view(&mut state, &mut host, mid).is_none(),
        "two non-adjacent pages must take the staged path this test is about"
    );

    // 2x2 BGRA8: 8 tight bytes per row at a 128-byte pitch, so 120 bytes of
    // guest content sit between the rows.
    let (w, h, bpp) = (2u32, 2u32, 4u32);
    let tight = (w * bpp) as usize;
    let bpr = 128u32;
    let span_end = (bpr as u64) * (h as u64 - 1) + tight as u64;
    let src = [
        0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        0x10,
    ];
    assert!(write_full_rect_raw_at(
        &mut state,
        &mut host,
        mid,
        0,
        bpr,
        span_end,
        w,
        h,
        bpp,
        &src,
        tight as u32,
    ));

    let mut got = vec![0u8; page as usize];
    assert!(host.read_gpa(gpa0, &mut got).is_ok());
    let mut want = vec![0xCCu8; page as usize];
    want[..tight].copy_from_slice(&src[..tight]);
    want[bpr as usize..bpr as usize + tight].copy_from_slice(&src[tight..]);
    let first_diff = got.iter().zip(want.iter()).position(|(a, b)| a != b);
    assert_eq!(
        first_diff, None,
        "byte {first_diff:?} outside the texel runs was modified"
    );
}

#[test]
fn write_bgra8_fragmented_skips_final_row_padding() {
    use crate::model::PAGE_SHIFT_X86;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    let gpa0 = 0x3500_0000u64;
    let gpa1 = 0x4600_0000u64;
    host.map_range(gpa0, page as usize, 0xCC);
    host.map_range(gpa1, page as usize, 0xCC);
    let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
    let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
    let mid = 13u32;
    state.map_surface(mid);
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![
            (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    let src = [
        0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        0x10,
    ];
    assert!(
        mapper::ensure_contig_view(&mut state, &mut host, mid).is_none(),
        "two non-adjacent pages must take the staged path this test is about"
    );
    assert!(write_bgra8(&mut state, &mut host, mid, &src, 8, 2, 2));

    // No device descriptor, so the invented window applies: tight = 2 × 4,
    // bpr = ALIGN_UP(8, ROW_BYTES_ALIGN) = 128, two rows — a pitch wider
    // than the packed row, so there is inter-row padding to get wrong.
    let tight = 8usize;
    let bpr = 128usize;
    let mut got = vec![0u8; page as usize];
    assert!(host.read_gpa(gpa0, &mut got).is_ok());
    let mut want = vec![0xCCu8; page as usize];
    want[..tight].copy_from_slice(&src[..tight]);
    want[bpr..bpr + tight].copy_from_slice(&src[tight..]);
    let first_diff = got.iter().zip(want.iter()).position(|(a, b)| a != b);
    assert_eq!(
        first_diff, None,
        "byte {first_diff:?} outside the sample window was modified"
    );
}

/// The packed-contig BGRA write pokes rows straight into a raw host pointer,
/// so its only bound is the sample window `contig_for_span` validated. The
/// fragmented path's equivalent is checked by
/// `write_bgra8_fragmented_skips_final_row_padding`; this pins the same
/// contract on the pointer path, where an overrun is a write into whatever
/// guest allocation follows rather than a refused import.
///
/// Asserted as "every byte outside the window is unchanged", not just the
/// final row's padding: inter-row padding belongs to the same class and a
/// stride bug hits it first.
#[test]
fn write_bgra8_contig_writes_only_inside_the_sample_window() {
    use crate::model::PAGE_SHIFT_X86;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    let gpa = 0x7300_0000u64;
    host.map_range(gpa, page as usize, 0xCC);
    let pfn = (gpa >> PAGE_SHIFT_X86) as u32;
    let mid = 21u32;
    state.map_surface(mid);
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    // No device descriptor, so the invented window applies: tight = 2 × 4,
    // bpr = ALIGN_UP(8, ROW_BYTES_ALIGN) = 128, two rows.
    let tight = 8usize;
    let bpr = 128usize;
    let src: Vec<u8> = (0..16u8).map(|i| i.wrapping_mul(17)).collect();
    assert!(
        mapper::ensure_contig_view(&mut state, &mut host, mid).is_some(),
        "one packed page must take the contig path this test is about"
    );
    assert!(write_bgra8(&mut state, &mut host, mid, &src, 8, 2, 2));

    let mut got = vec![0u8; page as usize];
    assert!(host.read_gpa(gpa, &mut got).is_ok());
    let mut want = vec![0xCCu8; page as usize];
    want[..tight].copy_from_slice(&src[..tight]);
    want[bpr..bpr + tight].copy_from_slice(&src[tight..]);
    let first_diff = got.iter().zip(want.iter()).position(|(a, b)| a != b);
    assert_eq!(
        first_diff, None,
        "byte {first_diff:?} outside the sample window was modified"
    );
}

/// Fragmented compute staging materializes the sample window once and
/// preserves padded-row addressing across non-contiguous guest pages.
#[test]
fn read_rect_raw_fragmented_pages_with_padded_rows() {
    use crate::model::PAGE_SHIFT_X86;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    let gpa0 = 0x5100_0000u64;
    let gpa1 = 0x6200_0000u64;
    host.map_range(gpa0, page as usize, 0);
    host.map_range(gpa1, page as usize, 0);
    let row0 = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let row1 = [9u8, 10, 11, 12, 13, 14, 15, 16];
    host.write_gpa(gpa0, &row0).unwrap();
    host.write_gpa(gpa1, &row1).unwrap();

    let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
    let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
    let mid = 12u32;
    state.map_surface(mid);
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![
            (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    let mut dst = [0u8; 16];
    assert!(read_rect_raw_at(
        &mut state,
        &mut host,
        mid,
        SurfaceWindow {
            base_off: 0,
            bpr: page as u32,
            span_end: page + row1.len() as u64,
            bpp: 4
        },
        Rect {
            origin_x: 0,
            origin_y: 0,
            width: 2,
            height: 2
        },
        &mut dst,
        8
    ));
    assert_eq!(&dst[..8], &row0);
    assert_eq!(&dst[8..], &row1);
}

/// A sub-rectangle of a padded plane over scattered guest pages reads through
/// one page-table walk, not through a plane-sized window.
///
/// This is the source half of every mapper-ref-texture to linear blit. Before the
/// rectangle shape reached this rail the arm below materialised the *whole*
/// sample window into a fresh zeroed `Vec` and then copied the wanted rows out
/// of it, so a rectangle covering a fraction of the plane still paid for all of
/// it twice. The fixture is deliberately a strict sub-rectangle in both axes
/// with a row stride wider than its rows, which is the shape the old arm could
/// not narrow and the old full-plane-tight special case could not claim.
#[test]
fn a_packed_sub_rectangle_of_a_scattered_plane_reads_through_one_walk() {
    use crate::model::PAGE_SHIFT_X86;
    use crate::runtime::drain::store_route_count;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    // Four pages, none adjacent, so the walk has to split into four runs.
    let gpas = [0x5100_0000u64, 0x6200_0000, 0x4300_0000, 0x7400_0000];
    let bpr = page as u32 / 2;
    let rows = 8u32;
    let bpp = 4u32;
    // Plane bytes, distinct per byte, laid into the pages the mapping names.
    let plane: Vec<u8> = (0..(bpr as u64 * rows as u64))
        .map(|i| (i % 251) as u8)
        .collect();
    for (i, gpa) in gpas.iter().enumerate() {
        host.map_range(*gpa, page as usize, 0);
        let lo = i * page as usize;
        host.write_gpa(*gpa, &plane[lo..lo + page as usize])
            .unwrap();
    }
    let mid = 21u32;
    state.map_surface(mid);
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = gpas
            .iter()
            .map(|gpa| {
                (((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID
            })
            .collect();
    }

    let origin_x = 3u32;
    let origin_y = 2u32;
    let width = 5u32;
    let height = 4u32;
    let rb = (width * bpp) as usize;
    let mut dst = vec![0u8; rb * height as usize];
    let walks = store_route_count("rectrd_rect_walk");
    let windows = store_route_count("rectrd_window_padded_dst");
    assert!(read_rect_raw_at(
        &mut state,
        &mut host,
        mid,
        SurfaceWindow {
            base_off: 0,
            bpr,
            span_end: bpr as u64 * rows as u64,
            bpp,
        },
        Rect {
            origin_x,
            origin_y,
            width,
            height,
        },
        &mut dst,
        rb as u32,
    ));
    assert_eq!(
        store_route_count("rectrd_rect_walk") - walks,
        1,
        "a packed destination must resolve the page table once for the rectangle"
    );
    assert_eq!(
        store_route_count("rectrd_window_padded_dst") - windows,
        0,
        "the plane-sized window arm is for padded destinations only"
    );
    for y in 0..height as usize {
        let src_off = (origin_y as usize + y) * bpr as usize + (origin_x * bpp) as usize;
        assert_eq!(
            &dst[y * rb..(y + 1) * rb],
            &plane[src_off..src_off + rb],
            "row {y} did not land at its texel offset"
        );
    }
}

/// A rect ending past the sample window must be refused the same way and
/// named the same way whichever arm reads it. The bound used to live inside
/// the contig arm, so the fragmented arm — the one a driven x86 boot takes —
/// answered an overrun with a bare `false` from a slice index and no line
/// saying the rect had left the window.
///
/// Run over both arms from one body so the two cannot drift: a single packed
/// page takes the contig arm, two scattered pages take the fragmented one.
#[test]
fn a_rect_past_the_sample_window_is_named_on_both_read_arms() {
    use crate::model::PAGE_SHIFT_X86;

    for scattered in [false, true] {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpa0 = 0x5100_0000u64;
        let gpa1 = 0x6200_0000u64;
        host.map_range(gpa0, page as usize, 0);
        host.map_range(gpa1, page as usize, 0);
        let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
        let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
        let mid = 12u32;
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            // One page is a packed view; two distant ones cannot be.
            m.page_entries = if scattered {
                vec![
                    (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                    (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                ]
            } else {
                vec![(pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID]
            };
        }
        assert_eq!(
            mapper::ensure_contig_view(&mut state, &mut host, mid).is_some(),
            !scattered,
            "scattered={scattered} must select the arm this iteration is about"
        );

        // The window is one page. Asking for four rows at a one-page pitch
        // puts the fourth row's last byte three pages past its end.
        let mut dst = [0u8; 4 * 8];
        let cap = crate::observe::FailCapture::start();
        let ok = read_rect_raw_at(
            &mut state,
            &mut host,
            mid,
            SurfaceWindow {
                base_off: 0,
                bpr: page as u32,
                span_end: page,
                bpp: 4,
            },
            Rect {
                origin_x: 0,
                origin_y: 0,
                width: 2,
                height: 4,
            },
            &mut dst,
            8,
        );
        let overruns: Vec<String> = cap
            .lines()
            .into_iter()
            .filter(|l| l.contains("reason=read_overrun"))
            .collect();
        assert!(!ok, "scattered={scattered}: the read must refuse");
        assert_eq!(
            overruns.len(),
            1,
            "scattered={scattered}: the refusal must name the bound it broke, \
             not leave the caller's decline to stand for it: {overruns:?}"
        );
    }
}

/// compute_full_tight_scratch: an exact-pitch fragmented compute plane
/// reads and writes directly through the caller's tight buffer. The
/// always-on proxy proves this class is selected on a live dispatch.
///
/// The read half is the rectangle walk and the write half still has its own
/// full-plane-tight arm, so the two proxies differ: a counter for the read, the
/// `full_tight_direct` line for the write. The read's separate special case was
/// retired because the rectangle subsumes it — a tight full plane is a
/// rectangle whose rows happen to touch, and it moves as one piece.
#[test]
fn fragmented_full_tight_rect_uses_direct_mapping_window() {
    use crate::model::PAGE_SHIFT_X86;
    use crate::runtime::drain::store_route_count;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    let gpa0 = 0x7100_0000u64;
    let gpa1 = 0x8200_0000u64;
    host.map_range(gpa0, page as usize, 0x31);
    host.map_range(gpa1, page as usize, 0x42);
    let mid = 29;
    assert!(state.map_surface(mid));
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![
            (((gpa0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            (((gpa1 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }

    let bpr = page as u32;
    let span = page * 2;
    let mut tight = vec![0u8; span as usize];
    let walks = store_route_count("rectrd_rect_walk");
    assert!(read_rect_raw_at(
        &mut state,
        &mut host,
        mid,
        SurfaceWindow {
            base_off: 0,
            bpr,
            span_end: span,
            bpp: 4
        },
        Rect {
            origin_x: 0,
            origin_y: 0,
            width: bpr / 4,
            height: 2
        },
        &mut tight,
        bpr
    ));
    assert!(tight[..page as usize].iter().all(|&v| v == 0x31));
    assert!(tight[page as usize..].iter().all(|&v| v == 0x42));
    assert_eq!(
        store_route_count("rectrd_rect_walk") - walks,
        1,
        "a tight full plane is a rectangle and must take the one-walk arm"
    );

    tight.fill(0x5a);
    assert!(write_full_rect_raw_at(
        &mut state,
        &mut host,
        mid,
        0,
        bpr,
        span,
        bpr / 4,
        2,
        4,
        &tight,
        bpr,
    ));
    let mut check = vec![0u8; span as usize];
    assert!(mapper::read_mapping_bytes(
        &mut state, &mut host, mid, 0, &mut check,
    ));
    assert_eq!(check, tight);

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(log.contains(&format!(
        "OFF mapping_write full_tight_direct mid={mid} bytes={span}"
    )));
}

/// A published descriptor that resolves no plane is not the same state as no
/// descriptor at all, and the device must not answer them alike.
///
/// Both used to return the packed window over offset 0. For the second that
/// is the only layout information anyone has; for the first the guest has
/// already said where its planes are and this texture matched two of them —
/// a v0a8 surface's Y and alpha planes share format and geometry, so the
/// scan cannot separate them and plane 0's bytes would be bound for a sample
/// the wire meant for alpha, silently.
#[test]
fn an_ambiguous_descriptor_declines_where_an_absent_one_still_sizes_a_window() {
    use crate::protocol::endian::{st16, st32, st64};
    use crate::protocol::iosurface_pages::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_PLANES, DEVICE_DESC_PLANE_COUNT, DEVICE_PLANE_BPE,
        DEVICE_PLANE_BPR, DEVICE_PLANE_DESC_LEN, DEVICE_PLANE_DIMS, DEVICE_PLANE_OFFSET,
        DEVICE_PLANE_SIZE,
    };
    use crate::protocol::pixel_format::MTL_FORMAT_R8_UNORM;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.map_surface(8);

    // No descriptor yet: geometry came from the mapper-ref-texture object and
    // the aligned row stands in for the pitch. 4 R8 texels align to 128.
    let m = state.mappings.get(&8).expect("mapping");
    assert_eq!(
        mapper_ref_texture_sample_window(m, 4, 2, MTL_FORMAT_R8_UNORM),
        Some((0, 128, 256)),
        "with nothing published there are no planes to confuse"
    );

    // Publish a v0a8-shaped descriptor: planes 0 and 2 are both R8 4x2.
    let mut desc = vec![0u8; crate::protocol::iosurface_pages::DEVICE_DESC_LEN];
    st32(&mut desc[DEVICE_DESC_ALLOC_SIZE..], 0x2000);
    desc[DEVICE_DESC_PLANE_COUNT] = 3;
    let pack = |w: u32, h: u32| ((w as u64 & 0xffffff) << 8) | ((h as u64 & 0xffffff) << 40);
    for (i, (off, w, h, bpe)) in [(512u32, 4u32, 2u32, 1u16), (1024, 2, 1, 2), (1536, 4, 2, 1)]
        .iter()
        .enumerate()
    {
        let p = DEVICE_DESC_PLANES + i * DEVICE_PLANE_DESC_LEN;
        st32(&mut desc[p + DEVICE_PLANE_OFFSET..], *off);
        st32(&mut desc[p + DEVICE_PLANE_SIZE..], 256);
        st64(&mut desc[p + DEVICE_PLANE_DIMS..], pack(*w, *h));
        st32(&mut desc[p + DEVICE_PLANE_BPR..], 64);
        st16(&mut desc[p + DEVICE_PLANE_BPE..], *bpe);
    }
    assert!(state.set_mapping_device_desc(8, &desc));

    let m = state.mappings.get(&8).expect("mapping");
    assert_eq!(
        mapper_ref_texture_sample_window(m, 4, 2, MTL_FORMAT_R8_UNORM),
        None,
        "two planes match and neither is the answer, so nothing is bound"
    );
    // The wire index is the only thing that separates them, and it reaches
    // each of the two directly.
    assert_eq!(
        ref_texture_sample_window(m, 0, 4, 2, MTL_FORMAT_R8_UNORM).map(|w| w.0),
        Some(512)
    );
    assert_eq!(
        ref_texture_sample_window(m, 2, 4, 2, MTL_FORMAT_R8_UNORM).map(|w| w.0),
        Some(1536)
    );
    // An index past the plane count resolves nothing rather than falling
    // back onto plane 0's bytes.
    assert_eq!(
        ref_texture_sample_window(m, 7, 4, 2, MTL_FORMAT_R8_UNORM),
        None
    );
}

/// qemu-shim: guest page write IS the surface content (unified memory) —
/// bytes land in pages and the generation advances; nothing else exists.
#[test]
fn write_bgra8_lands_in_pages_and_bumps_gen() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x18u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(8);
    {
        let m = state.mappings.get_mut(&8).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
    }
    assert!(state.set_mapping_geom(8, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    // BGRA red pixel + zeros
    let src = [0x00u8, 0x00, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    assert!(write_bgra8(&mut state, &mut host, 8, &src, 8, 2, 2));
    let m = state.mappings.get(&8).unwrap();
    assert_eq!(m.content_generation, 1);
    let mut first_px = [0u8; 4];
    assert!(host.read_gpa(gpa, &mut first_px).is_ok());
    assert_eq!(&first_px, &[0x00, 0x00, 0xFF, 0xFF], "pages hold the write");
}

#[test]
fn raw_rows_roundtrip() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x11u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(4);
    let m = state.mappings.get_mut(&4).unwrap();
    m.mapped = true;
    m.mapping_internal = 1;
    m.page_entries = vec![entry];
    assert!(state.set_mapping_geom(4, 2, 2, 0));
    // 2x2 depth32 floats: 1.0, 0.5 / 0.25, 0.0
    let mut src = Vec::new();
    for f in [1.0f32, 0.5, 0.25, 0.0] {
        src.extend_from_slice(&f.to_bits().to_le_bytes());
    }
    assert!(write_raw_rows(&mut state, &mut host, 4, &src, 8, 8, 2, 2));
    let gen = state.mappings.get(&4).unwrap().content_generation;
    assert!(gen >= 1);
    let mut dst = vec![0u8; 16];
    assert!(read_raw_rows(
        &mut state, &mut host, 4, &mut dst, 8, 8, 2, 2
    ));
    assert_eq!(dst, src);
    // Read does not bump generation.
    assert_eq!(state.mappings.get(&4).unwrap().content_generation, gen);
}

/// The read side of the same bound. A rect read whose geometry exceeds what
/// `span_end` allows must be REJECTED, not run past the contig view.
///
/// `contig_for_span` guarantees the view covers `span_end` and nothing more,
/// so an oversized `height` reads whatever is next in the QEMU process —
/// unrelated memory sampled into a texture, or a SIGSEGV that takes the VM
/// down with no guest-side trace. The write side has carried this guard for a
/// while; the read side did not, which is the asymmetry to watch for when a
/// raw-pointer fast path is added beside a checked slow path.
///
/// A correctly-sized read (read_end == span_end) still succeeds.
#[test]
fn oversized_height_rect_read_is_rejected_not_overrun() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x23u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    // A full 16 KiB page, so `contig_for_span` succeeds and the guard — not
    // the view length — is what has to stop the overrun.
    host.map_range(gpa, 0x4000, 0xCC);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(11);
    {
        let m = state.mappings.get_mut(&11).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
    }
    // The source allows exactly 2 rows of bpr=8.
    let bpr = 8u32;
    let (width, bpp) = (2u32, 4u32); // row_bytes = 8 == bpr (dense path)
    let span_end = 16u64;

    // 100 rows: read_end = (100-1)*8 + 8 = 800 > 16.
    let mut big = vec![0u8; 100 * bpr as usize];
    let cap = crate::observe::FailCapture::start();
    assert!(
        !read_rect_raw_at(
            &mut state,
            &mut host,
            11,
            SurfaceWindow {
                base_off: 0,
                bpr,
                span_end,
                bpp
            },
            Rect {
                origin_x: 0,
                origin_y: 0,
                width,
                height: 100
            },
            &mut big,
            bpr
        ),
        "an oversized-height read must be rejected"
    );
    assert!(
        cap.one("mapping_read").contains("reason=read_overrun"),
        "the refusal must name itself"
    );
    assert!(
        big.iter().all(|&b| b == 0),
        "a rejected read must not have copied anything into the caller's buffer"
    );
    drop(cap);

    // A correctly-sized 2-row read (read_end == span_end) still succeeds.
    let mut ok = vec![0u8; 2 * bpr as usize];
    assert!(
        read_rect_raw_at(
            &mut state,
            &mut host,
            11,
            SurfaceWindow {
                base_off: 0,
                bpr,
                span_end,
                bpp
            },
            Rect {
                origin_x: 0,
                origin_y: 0,
                width,
                height: 2
            },
            &mut ok,
            bpr
        ),
        "a read whose extent equals span_end must succeed"
    );
    assert_eq!(ok, vec![0xCC; 2 * bpr as usize], "and must read the page");
}

/// A writeback whose source `height` exceeds what the destination `span_end`
/// allows must be REJECTED, not run past the contig view into adjacent guest
/// pages (the trace-less heap smash behind).
/// A correctly-sized write (write_end == span_end) still succeeds.
#[test]
fn oversized_height_writeback_is_rejected_not_overrun() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x21u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    // Map a full 16 KiB page so contig_for_span succeeds; the guard, not the
    // view length, must be what stops the overrun.
    host.map_range(gpa, 0x4000, 0xCC); // 0xCC canary fills the page
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(9);
    {
        let m = state.mappings.get_mut(&9).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
    }
    // Destination allows exactly 2 rows of bpr=8 (span_end = 2*8 = 16).
    let bpr = 8u32;
    let (width, bpp) = (2u32, 4u32); // row_bytes = 8 == bpr (dense path)
    let span_end = 16u64;
    // Oversized source: 100 rows. write_end = (100-1)*8 + 8 = 800 > 16.
    let big = vec![0x2a; 100 * bpr as usize];
    assert!(
        !write_full_rect_raw_at(
            &mut state, &mut host, 9, 0, bpr, span_end, width, 100, bpp, &big, bpr,
        ),
        "an oversized-height write must be rejected"
    );
    // Nothing past span_end was written — the canary survives at offset 100.
    let mut probe = [0u8; 4];
    assert!(mapper::read_mapping_bytes(
        &mut state, &mut host, 9, 100, &mut probe
    ));
    assert_eq!(
        probe, [0xCC; 4],
        "guest bytes past span_end must be untouched"
    );
    // A correctly-sized 2-row write (write_end == span_end) still succeeds.
    let ok = vec![0x2a; 2 * bpr as usize];
    assert!(
        write_full_rect_raw_at(
            &mut state, &mut host, 9, 0, bpr, span_end, width, 2, bpp, &ok, bpr,
        ),
        "a write whose extent equals span_end must succeed"
    );
}

/// Clear+partial Store: seed=None (full write) must overwrite prior guest
/// content outside the scissor — logo-mid residual when seed=clear skipped.
#[test]
fn clear_store_full_write_overwrites_prior_guest_outside_scissor() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x14u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(7);
    let m = state.mappings.get_mut(&7).unwrap();
    m.mapped = true;
    m.mapping_internal = 1;
    m.page_entries = vec![entry];
    // 4x2 BGRA
    assert!(state.set_mapping_geom(7, 4, 2, MTL_FORMAT_BGRA8_UNORM));
    // Prior guest content: "logo" non-zero all pixels.
    let mut logo = vec![0u8; 4 * 2 * 4];
    for px in logo.chunks_exact_mut(4) {
        px.copy_from_slice(&[0x10, 0x20, 0x30, 0xFF]); // BGRA
    }
    assert!(write_bgra8(&mut state, &mut host, 7, &logo, 16, 4, 2));
    // Metal RT after Clear+partial toolbar: clear everywhere, one red pixel
    // at (1,0) as the drawn strip. Full Store (seed=None).
    let mut rgba = vec![0u8; 4 * 2 * 4]; // clear = zeros RGBA
    rgba[4] = 255; // R
    rgba[4 + 3] = 255; // A
    assert!(write_rgba8_image_changed(
        &mut state,
        &mut host,
        7,
        &rgba,
        None, // Clear Store: not image_changed vs clear seed
        4,
        2,
        FramePublication::HostCache
    ));
    let mut row = vec![0u8; 16];
    assert!(read_rect_raw(
        &mut state,
        &mut host,
        7,
        Rect {
            origin_x: 0,
            origin_y: 0,
            width: 4,
            height: 1
        },
        &mut row,
        16
    ));
    // Outside scissor pixel 0 must be clear (not logo).
    assert_eq!(
        &row[0..4],
        &[0, 0, 0, 0],
        "Clear Store must wipe prior guest"
    );
    // Drawn pixel 1 red in BGRA.
    assert_eq!(&row[4..8], &[0, 0, 255, 255]);
    // Contrast: Load seed=logo + same rgba would leave logo where equal —
    // not tested here; store_seed_policy gates that path.
}

/// The depth/stencil writeback must name its refusals too.
///
/// `write_raw_rows` is the third guest-memory writer in this file and was
/// the last one still answering every refusal with a bare `false`. It is
/// worse placed than the others to be silent: both callers discard its
/// result outright, so nothing above it could report a reason even if it
/// wanted to, and the guest work it drops is a `MTLStoreActionStore` on a
/// depth/stencil attachment - the mapping simply keeps stale bytes and the
/// pass reports success. The colour writeback twenty lines from its caller
/// emits for the analogous condition.
#[test]
fn every_raw_rows_refusal_names_itself() {
    use crate::runtime::drain::store_route_count;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x14u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(7);
    let m = state.mappings.get_mut(&7).unwrap();
    m.mapped = true;
    m.mapping_internal = 1;
    m.page_entries = vec![entry];
    assert!(state.set_mapping_geom(7, 4, 2, MTL_FORMAT_BGRA8_UNORM));
    let rows = vec![0u8; 4 * 2 * 4];

    // A zero dimension is not a rect.
    let n = store_route_count("surface_write_geometry");
    assert!(!write_raw_rows(
        &mut state, &mut host, 7, &rows, 16, 16, 0, 2
    ));
    assert_eq!(store_route_count("surface_write_geometry"), n + 1);

    // A source pitch that cannot hold one row.
    let n = store_route_count("surface_write_source_stride");
    assert!(!write_raw_rows(
        &mut state, &mut host, 7, &rows, 4, 16, 4, 2
    ));
    assert_eq!(store_route_count("surface_write_source_stride"), n + 1);

    // The source ends before the rows it declares.
    let n = store_route_count("surface_write_source_short");
    assert!(!write_raw_rows(
        &mut state,
        &mut host,
        7,
        &rows[..8],
        16,
        16,
        4,
        2
    ));
    assert_eq!(store_route_count("surface_write_source_short"), n + 1);

    // No such mapping.
    let n = store_route_count("surface_write_mapping_absent");
    assert!(!write_raw_rows(
        &mut state, &mut host, 4242, &rows, 16, 16, 4, 2
    ));
    assert_eq!(store_route_count("surface_write_mapping_absent"), n + 1);

    // The latched geometry is not this frame's.
    let n = store_route_count("surface_write_geometry_moved");
    let big = vec![0u8; 8 * 8 * 4];
    assert!(!write_raw_rows(
        &mut state, &mut host, 7, &big, 32, 32, 8, 8
    ));
    assert_eq!(store_route_count("surface_write_geometry_moved"), n + 1);

    // Unmapped: there is nowhere to write.
    let n = store_route_count("surface_write_mapping_not_resident");
    state.mappings.get_mut(&7).unwrap().mapped = false;
    assert!(!write_raw_rows(
        &mut state, &mut host, 7, &rows, 16, 16, 4, 2
    ));
    assert_eq!(
        store_route_count("surface_write_mapping_not_resident"),
        n + 1
    );
}

/// Every refusal in `write_rgba8_image_changed` must name itself.
///
/// This writeback is the guest's own copy of a rendered frame, and it is
/// reached on the live x86/Vulkan sync-store route. Every one of its
/// refusals used to be a bare `false`, so a frame the guest never received
/// left no trace at all — while the sibling writer in this same file
/// answered the identical conditions through `SurfaceWriteRefusal`. The
/// vocabulary was already complete; only this arm did not use it.
///
/// Asserting on the route slugs rather than on the fail lines is what makes
/// this a regression test: `refuse` latches its line per `(check, mapping)`
/// but always counts, so a reverted arm shows up as a counter that stops
/// moving even on a mapping that has already refused once.
#[test]
fn every_rgba8_image_changed_refusal_names_itself() {
    use crate::runtime::drain::store_route_count;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x14u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(7);
    let m = state.mappings.get_mut(&7).unwrap();
    m.mapped = true;
    m.mapping_internal = 1;
    m.page_entries = vec![entry];
    assert!(state.set_mapping_geom(7, 4, 2, MTL_FORMAT_BGRA8_UNORM));
    let frame = vec![0u8; 4 * 2 * 4];

    // Each case: (slug, the call that must take that arm).
    let before = |slug: &str| store_route_count(slug);

    // A zero dimension is not a rect.
    let n = before("surface_write_geometry");
    assert!(!write_rgba8_image_changed(
        &mut state,
        &mut host,
        7,
        &frame,
        None,
        0,
        2,
        FramePublication::HostCache
    ));
    assert_eq!(store_route_count("surface_write_geometry"), n + 1);

    // The source ends before the frame it declares.
    let n = before("surface_write_source_short");
    assert!(!write_rgba8_image_changed(
        &mut state,
        &mut host,
        7,
        &frame[..4],
        None,
        4,
        2,
        FramePublication::HostCache
    ));
    assert_eq!(store_route_count("surface_write_source_short"), n + 1);

    // The seed ends before it: a different buffer, so a different slug.
    let n = before("surface_write_seed_short");
    assert!(!write_rgba8_image_changed(
        &mut state,
        &mut host,
        7,
        &frame,
        Some(&frame[..4]),
        4,
        2,
        FramePublication::HostCache
    ));
    assert_eq!(
        store_route_count("surface_write_seed_short"),
        n + 1,
        "a short seed must not be reported as a short source"
    );

    // No such mapping: the surface went away between the arm and the landing.
    let n = before("surface_write_mapping_absent");
    assert!(!write_rgba8_image_changed(
        &mut state,
        &mut host,
        4242,
        &frame,
        None,
        4,
        2,
        FramePublication::HostCache
    ));
    assert_eq!(store_route_count("surface_write_mapping_absent"), n + 1);

    // The latched geometry is not the frame's: landing it would skew.
    let n = before("surface_write_geometry_moved");
    let big = vec![0u8; 8 * 8 * 4];
    assert!(!write_rgba8_image_changed(
        &mut state,
        &mut host,
        7,
        &big,
        None,
        8,
        8,
        FramePublication::HostCache
    ));
    assert_eq!(store_route_count("surface_write_geometry_moved"), n + 1);

    // Unmapped: there is nowhere to write.
    let n = before("surface_write_mapping_not_resident");
    state.mappings.get_mut(&7).unwrap().mapped = false;
    assert!(!write_rgba8_image_changed(
        &mut state,
        &mut host,
        7,
        &frame,
        None,
        4,
        2,
        FramePublication::HostCache
    ));
    assert_eq!(
        store_route_count("surface_write_mapping_not_resident"),
        n + 1
    );
}

#[test]
fn rgba8_image_changed_writes_only_diff_spans() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    // 4x2 BGRA: invent bpr 128 → one page.
    let pfn = 0x13u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(6);
    let m = state.mappings.get_mut(&6).unwrap();
    m.mapped = true;
    m.mapping_internal = 1;
    m.page_entries = vec![entry];
    assert!(state.set_mapping_geom(6, 4, 2, MTL_FORMAT_BGRA8_UNORM));
    // Seed: all zeros.
    let seed = vec![0u8; 4 * 2 * 4];
    // Image: one red pixel at (1,0), rest zero.
    let mut rgba = seed.clone();
    rgba[4] = 255; // R
    rgba[4 + 3] = 255; // A
    assert!(write_rgba8_image_changed(
        &mut state,
        &mut host,
        6,
        &rgba,
        Some(&seed),
        4,
        2,
        FramePublication::HostCache
    ));
    // Read back first row of mapping (BGRA native).
    let mut row = vec![0u8; 16];
    assert!(read_rect_raw(
        &mut state,
        &mut host,
        6,
        Rect {
            origin_x: 0,
            origin_y: 0,
            width: 4,
            height: 1
        },
        &mut row,
        16
    ));
    // Pixel 1 is red in BGRA: B=0 G=0 R=255 A=255
    assert_eq!(&row[4..8], &[0, 0, 255, 255]);
    assert_eq!(&row[0..4], &[0, 0, 0, 0]);
}

/// Every writer of a mapping's guest pages leaves *no* host-side copy claiming
/// to be them.
///
/// This is the rule the two rect writers and the raw-row writer used to break,
/// and it is not a rule about caches — it is what makes the hypervisor's witness
/// answerable. `mapping_guest_write_verdict` reports whether the *guest* wrote
/// these pages, and this device's own writes go through a mapped host pointer
/// that cannot set `DIRTY_MEMORY_VGA`, so the witness is structurally blind to
/// them. Only the writer can retire the copy, and a writer that does not leaves
/// a stale frame the witness will go on calling `Clean` forever.
///
/// The consequence was measured on the colour LOAD seed:
/// `draw::seed_from_published_surface` serves that entry as the attachment's
/// prior content, the pass composites onto it, and its Store publishes the
/// result back over the guest's pages — so a partial store's pixels are lost and
/// then *held*, which is the class the door's strict evidence standard exists to
/// prevent and could not prevent alone.
///
/// Every writer in this file is asserted, including the ones that already
/// published or forgot, because the value of the rule is that it has no
/// exceptions to remember.
/// The `cache_rows_are_native` shortcut's premise: for a BGRA8 mapping, this
/// writer's native row *is* the host cache's row, byte for byte.
///
/// `write_rgba8_image_changed` used to convert the frame twice — once per row
/// into the mapping's native texels for the guest write, and once whole-frame
/// into BGRA for the cache — and the second pass was 45.9 % of the Metal rail's
/// `store_us`. It now converts once and the row loop reads the cache. That is
/// only sound while these two produce identical bytes, so the equality is
/// asserted directly rather than left to the reader of two loops in different
/// parts of the file.
///
/// The sRGB spelling is included because it is the other format the shortcut
/// admits, and a transfer function applied on one side and not the other would
/// be invisible in any test that used only one of them.
///
/// The store rail is asked by ordinal rather than named as `Rgba8ToRow::Bgra8`
/// on purpose: the shortcut's condition is a `matches!` on those two ordinals,
/// so what has to hold is that *the ordinals the shortcut admits* reach the
/// swapping arm. Naming the variant would assert the swap and skip the mapping,
/// which is the half that can drift.
#[test]
fn a_bgra8_native_row_is_the_host_cache_row() {
    const W: u32 = 4;
    let rgba: Vec<u8> = (0..W * 4)
        .map(|i| (i as u8).wrapping_mul(7).wrapping_add(3))
        .collect();
    // What the cache build performs: RGBA8 in, BGRA8 out.
    let mut cache_row = vec![0u8; (W * 4) as usize];
    for x in 0..W as usize {
        let i = x * 4;
        cache_row[i] = rgba[i + 2];
        cache_row[i + 1] = rgba[i + 1];
        cache_row[i + 2] = rgba[i];
        cache_row[i + 3] = rgba[i + 3];
    }
    for format in [
        MTL_FORMAT_BGRA8_UNORM,
        crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB,
    ] {
        let mut native = vec![0u8; (W * 4) as usize];
        let rail = crate::protocol::pixel_format::Rgba8ToRow::for_format(format)
            .expect("both BGRA8 spellings have a store arm");
        assert!(rail.convert(&rgba, W, &mut native));
        assert_eq!(
            native, cache_row,
            "format {format:#x} takes the shortcut, so its native row must be the cache row"
        );
        assert_eq!(
            crate::protocol::pixel_format::tight_row_bytes(W, format),
            Some(W * 4),
            "the shortcut also requires the tight row to be the whole of the cache row"
        );
    }
}

/// A ceded write lands the guest's pages exactly as a cached one does, names
/// the frame, and holds none of its bytes.
///
/// The three clauses are the whole of [`FramePublication`]'s contract, and only
/// the first is checkable from the guest's side. The other two are what the rail
/// depends on: a cession that failed to name the frame would retire the caller's
/// resident content claim at the moment it earned it, and a cession that still
/// held bytes would serve them beside a resident that had superseded them.
#[test]
fn a_ceded_write_lands_the_same_guest_pages_and_publishes_no_host_bytes() {
    const W: u32 = 4;
    const H: u32 = 4;

    let frame: Vec<u8> = (0..(W * H * 4) as u8).collect();
    let land = |publication| {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let pfn = 0x12u32;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        state.map_surface(5);
        let m = state.mappings.get_mut(&5).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        assert!(state.set_mapping_geom(5, W, H, MTL_FORMAT_BGRA8_UNORM));
        assert!(write_rgba8_image_changed(
            &mut state,
            &mut host,
            5,
            &frame,
            None,
            W,
            H,
            publication,
        ));
        let mut pages = vec![0u8; (W * H * 4) as usize];
        assert!(read_rect_raw(
            &mut state,
            &mut host,
            5,
            Rect {
                origin_x: 0,
                origin_y: 0,
                width: W,
                height: H,
            },
            &mut pages,
            W * 4,
        ));
        (state, pages)
    };

    let (cached, cached_pages) = land(FramePublication::HostCache);
    let (ceded, ceded_pages) = land(FramePublication::RailResident);

    assert_eq!(
        ceded_pages, cached_pages,
        "the guest sees the same frame either way; the publication decides only \
         where this device keeps its own copy"
    );
    assert!(
        crate::runtime::surface_cache::get(&cached, 5, W, H).is_some(),
        "the cached arm publishes the bytes"
    );
    assert!(
        crate::runtime::surface_cache::get(&ceded, 5, W, H).is_none()
            && crate::runtime::surface_cache::surface_ceded_to_resident(&ceded, 5, W, H),
        "the ceded arm publishes no bytes, and says so as a cession rather than \
         as an absence"
    );
    assert!(
        crate::runtime::surface_cache::frame_generation(&ceded, 5, W, H).is_some(),
        "a ceded frame is still this mapping's published frame, which is what \
         the caller's resident content claim is compared against"
    );
}

/// A changed-span write publishes the *whole* frame to the host cache, even
/// though it lands only the rows that differ in the guest's pages.
///
/// This is what `take_frame_buffer`'s contract is for. The buffer it hands back
/// holds the previous frame, not zeroes, so a cache build that skipped a row —
/// for instance by following the row loop's own skip — would republish that
/// row's *old* pixels under a fresh generation. Every reader of this cache,
/// including the Metal rail's LOAD seed and its sampled rung, would then serve
/// them as current.
///
/// The fixture makes that failure visible: the second frame differs from the
/// first in exactly one row, so a whole-frame cache and a skipping one differ in
/// three of the four rows.
#[test]
fn a_changed_span_write_republishes_the_whole_frame_to_the_host_cache() {
    const W: u32 = 4;
    const H: u32 = 4;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x12u32;
    host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
    state.map_surface(5);
    let m = state.mappings.get_mut(&5).unwrap();
    m.mapped = true;
    m.mapping_internal = 1;
    m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    assert!(state.set_mapping_geom(5, W, H, MTL_FORMAT_BGRA8_UNORM));

    let first: Vec<u8> = [0x10u8, 0x20, 0x30, 0xff].repeat((W * H) as usize);
    assert!(write_rgba8_image_changed(
        &mut state,
        &mut host,
        5,
        &first,
        None,
        W,
        H,
        FramePublication::HostCache
    ));
    assert_eq!(
        crate::runtime::surface_cache::get(&state, 5, W, H).map(|f| f[..4].to_vec()),
        // RGBA 10/20/30 stored as BGRA.
        Some(vec![0x30, 0x20, 0x10, 0xff])
    );

    // The second frame differs from the first in row 2 only, and is handed the
    // first as its seed — so the guest write lands one row and the cache must
    // still carry four.
    let mut second = first.clone();
    let row = (W * 4) as usize;
    second[2 * row..3 * row].copy_from_slice(&[0x77u8, 0x88, 0x99, 0xff].repeat(W as usize));
    assert!(write_rgba8_image_changed(
        &mut state,
        &mut host,
        5,
        &second,
        Some(&first),
        W,
        H,
        FramePublication::HostCache
    ));

    let published = crate::runtime::surface_cache::get(&state, 5, W, H)
        .expect("a successful write publishes its frame")
        .to_vec();
    let mut expect = vec![0u8; second.len()];
    for x in 0..(W * H) as usize {
        let i = x * 4;
        expect[i] = second[i + 2];
        expect[i + 1] = second[i + 1];
        expect[i + 2] = second[i];
        expect[i + 3] = second[i + 3];
    }
    assert_eq!(
        published, expect,
        "the cache carries the whole frame, not only the rows the guest write touched"
    );
}

#[test]
fn no_writer_of_a_mappings_pages_leaves_a_host_copy_claiming_to_be_them() {
    const W: u32 = 4;
    const H: u32 = 2;

    /// A one-page BGRA mapping with a stale host-cache frame already in it, and
    /// a stamp saying nothing has written since — the state a full Store leaves
    /// and every writer below then invalidates.
    fn staged() -> (DeviceState, FakeHost) {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let pfn = 0x12u32;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        state.map_surface(5);
        let m = state.mappings.get_mut(&5).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        assert!(state.set_mapping_geom(5, W, H, MTL_FORMAT_BGRA8_UNORM));
        crate::runtime::surface_cache::store(
            &mut state,
            5,
            W,
            H,
            vec![0xCCu8; (W * H * 4) as usize],
        );
        crate::runtime::mapper::stamp_guest_write_gen(&mut state, &mut host, 5);
        assert!(
            crate::runtime::surface_cache::get(&state, 5, W, H).is_some(),
            "the fixture must start with a host copy for the writer to retire"
        );
        (state, host)
    }

    fn assert_retired(state: &DeviceState, who: &str) {
        assert!(
            crate::runtime::surface_cache::get(state, 5, W, H).is_none(),
            "{who} wrote the mapping's pages and left a host copy still claiming \
             to be them; the hypervisor's witness cannot see this device's own \
             writes, so nothing below would ever correct it"
        );
    }

    // A sub-rect: the partial store the Metal draw rail takes for a
    // single-scissor draw.
    let (mut state, mut host) = staged();
    let src = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    assert!(write_rect_raw(
        &mut state,
        &mut host,
        5,
        Rect {
            origin_x: 1,
            origin_y: 1,
            width: 2,
            height: 1
        },
        &src,
        8
    ));
    assert_retired(&state, "write_rect_raw");

    // The whole plane through the same writer, which is the blit arm — and the
    // worse case, because it replaces every texel the stale entry claims.
    let (mut state, mut host) = staged();
    let full = vec![0x77u8; (W * H * 4) as usize];
    assert!(write_rect_raw(
        &mut state,
        &mut host,
        5,
        Rect {
            origin_x: 0,
            origin_y: 0,
            width: W,
            height: H
        },
        &full,
        W * 4
    ));
    assert_retired(&state, "write_rect_raw over the whole plane");

    // Raw rows: depth and stencil, so the entry is not normally there at all —
    // asserted because the rule has no exceptions to remember.
    let (mut state, mut host) = staged();
    assert!(write_raw_rows(
        &mut state,
        &mut host,
        5,
        &vec![0x99u8; (W * H * 4) as usize],
        W * 4,
        W * 4,
        W,
        H
    ));
    assert_retired(&state, "write_raw_rows");

    // Native texels, which already carried the rule and is where the other two
    // took it from.
    let (mut state, mut host) = staged();
    assert!(write_native_image(
        &mut state,
        &mut host,
        5,
        &vec![0x55u8; (W * H * 4) as usize],
        W * 4,
        W,
        H,
        MTL_FORMAT_BGRA8_UNORM
    ));
    assert_retired(&state, "write_native_image");

    // The two that *publish* rather than retire satisfy the rule the other way:
    // the copy they leave is the frame they just wrote, so it does claim to be
    // the pages and is right to.
    let (mut state, mut host) = staged();
    assert!(write_bgra8(
        &mut state,
        &mut host,
        5,
        &vec![0x31u8; (W * H * 4) as usize],
        W * 4,
        W,
        H
    ));
    assert_eq!(
        crate::runtime::surface_cache::get(&state, 5, W, H).map(|f| f[0]),
        Some(0x31),
        "write_bgra8 publishes what it wrote, which is the other lawful outcome"
    );
}

#[test]
fn rect_raw_roundtrip_subregion() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    // 4x2 BGRA needs 4*4=16 tight, aligned bpr = 128 (ROW_BYTES_ALIGN).
    // One page is enough for 2 rows of 128.
    let pfn = 0x12u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(5);
    let m = state.mappings.get_mut(&5).unwrap();
    m.mapped = true;
    m.mapping_internal = 1;
    m.page_entries = vec![entry];
    assert!(state.set_mapping_geom(5, 4, 2, MTL_FORMAT_BGRA8_UNORM));
    // Write a 2x1 rect at (1,1): two BGRA pixels.
    let src = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    assert!(write_rect_raw(
        &mut state,
        &mut host,
        5,
        Rect {
            origin_x: 1,
            origin_y: 1,
            width: 2,
            height: 1
        },
        &src,
        8
    ));
    let mut dst = [0u8; 8];
    assert!(read_rect_raw(
        &mut state,
        &mut host,
        5,
        Rect {
            origin_x: 1,
            origin_y: 1,
            width: 2,
            height: 1
        },
        &mut dst,
        8
    ));
    assert_eq!(dst, src);
    // OOB origin fails.
    assert!(!write_rect_raw(
        &mut state,
        &mut host,
        5,
        Rect {
            origin_x: 3,
            origin_y: 0,
            width: 2,
            height: 1
        },
        &src,
        8
    ));
}
