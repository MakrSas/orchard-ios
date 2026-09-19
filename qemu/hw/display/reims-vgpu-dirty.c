/*
 * reims-vgpu: guest-write tracking over the hypervisor dirty bitmap.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/thread.h"
#include "qemu/rcu.h"
#include "exec/target_page.h"
#include "system/address-spaces.h"
#include "system/memory.h"
#include "system/ram_addr.h"
#include "system/physmem.h"

#include "reims-vgpu-dirty.h"

/*
 * One guest-RAM range of the system address space. A tracked page is resolved
 * against this table by address, so one entry serves every page that lands in
 * it — the same reason reims_vgpu_*_map_pages walks maximal runs.
 */
typedef struct ReimsVgpuDirtySlice {
    MemoryRegion *mr;
    uint64_t gpa;           /* first GPA covered */
    uint64_t len;           /* bytes covered */
    ram_addr_t ram_addr;    /* ram_addr_t of `gpa` */
    hwaddr offset;          /* offset of `gpa` within mr */
    /* Whether this device was already recording writes to the range when this
     * harvest's sync ran, and whether a tracked page asked it to start. Held
     * per slice rather than in parallel arrays because they are read in the
     * page loop, which is the harvest's inner loop. */
    bool logged;
    bool needs_log;
} ReimsVgpuDirtySlice;

/* A tracked page set: one mapping incarnation's guest storage. */
typedef struct ReimsVgpuDirtySet {
    uint64_t *pages;        /* sorted, deduplicated, page-aligned */
    /*
     * Parallel to `pages`: the value `gen` took at the harvest that last saw
     * this page written, or 0 for a page never seen written. A set-level
     * generation can only say "something in here moved", which forces a reader
     * holding a whole-surface copy to discard all of it; this is what lets it
     * discard exactly the pages that moved.
     */
    uint64_t *page_gen;
    size_t count;
    uint64_t page_size;
    uint64_t gen;           /* 0 until armed */
    uint64_t arm_at;        /* harvest count at which gen may leave 0 */
} ReimsVgpuDirtySet;

struct ReimsVgpuDirty {
    QemuMutex lock;
    GHashTable *sets;       /* token -> ReimsVgpuDirtySet* */
    uint64_t next_token;
    /*
     * Regions this device turned DIRTY_MEMORY_VGA logging on for, referenced
     * so the pointers stay valid until we turn it back off. Logging stays on
     * for the device's life: toggling it per surface re-protects the whole
     * region each time, and the guest's entire working set would refault for a
     * saving this device never collects.
     */
    GHashTable *logged;     /* MemoryRegion* -> itself */
    /*
     * The guest-RAM ranges of the system address space, rebuilt at the top of
     * each harvest and owned across harvests only so the steady state does not
     * allocate. Grown to whatever the flat view reports and never bounded: a
     * range this table dropped would read as written for every page in it, for
     * the life of the VM, with nothing to say so — which is precisely the
     * failure the flat view walk replaced.
     *
     * Measured at 5 ranges on x86 q35 with this device, unchanged across a
     * driven boot including the load phase. So this is not a table under
     * pressure, and the growth is not there to serve a reach anyone has seen.
     * It is here because the alternative to growing is dropping, and dropping
     * is unobservable from the guest side and permanent.
     */
    ReimsVgpuDirtySlice *slices;
    int slices_cap;
    uint64_t harvests;
    /*
     * Generation reads since the last harvest. A harvest nothing has consumed
     * since the previous one cannot tell any reader something it does not
     * already know, so it is skipped — which collapses a burst of doorbell
     * writes into one sync.
     */
    uint64_t reads_since_harvest;
};

static void reims_vgpu_dirty_set_free(gpointer p)
{
    ReimsVgpuDirtySet *s = p;

    g_free(s->pages);
    g_free(s->page_gen);
    g_free(s);
}

ReimsVgpuDirty *reims_vgpu_dirty_new(void)
{
    ReimsVgpuDirty *d = g_new0(ReimsVgpuDirty, 1);

    qemu_mutex_init(&d->lock);
    d->sets = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free,
                                    reims_vgpu_dirty_set_free);
    d->logged = g_hash_table_new(NULL, NULL);
    return d;
}

void reims_vgpu_dirty_free(ReimsVgpuDirty *d)
{
    GHashTableIter it;
    gpointer key, val;

    if (!d) {
        return;
    }
    /* BQL: memory_region_set_log is a MemoryRegion transaction. */
    g_hash_table_iter_init(&it, d->logged);
    while (g_hash_table_iter_next(&it, &key, &val)) {
        MemoryRegion *mr = key;

        memory_region_set_log(mr, false, DIRTY_MEMORY_VGA);
        memory_region_unref(mr);
    }
    g_hash_table_destroy(d->logged);
    g_hash_table_destroy(d->sets);
    g_free(d->slices);
    qemu_mutex_destroy(&d->lock);
    g_free(d);
}

static int reims_vgpu_dirty_cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;

    return x < y ? -1 : (x > y ? 1 : 0);
}

/*
 * One guest page a harvest found written, with the geometry it was tracked at.
 * The size travels with the page because the clear must cover the whole guest
 * page and the two pathways do not share a page shift.
 */
typedef struct ReimsVgpuDirtyWritten {
    uint64_t gpa;
    uint64_t page_size;
} ReimsVgpuDirtyWritten;

static int reims_vgpu_dirty_cmp_written(gconstpointer a, gconstpointer b)
{
    const ReimsVgpuDirtyWritten *x = a;
    const ReimsVgpuDirtyWritten *y = b;

    if (x->gpa != y->gpa) {
        return x->gpa < y->gpa ? -1 : 1;
    }
    /* Larger page first, so a duplicate of the same base collapses into the
     * run the wider page opens rather than truncating it. */
    return x->page_size > y->page_size ? -1 : (x->page_size < y->page_size);
}

uint64_t reims_vgpu_dirty_track(ReimsVgpuDirty *d, const uint64_t *gpas,
                                size_t count, size_t page_size)
{
    ReimsVgpuDirtySet *s;
    uint64_t *pages;
    uint64_t token;
    size_t i, n;

    if (!d || !gpas || count == 0 || page_size == 0 ||
        (page_size & (page_size - 1)) != 0) {
        return 0;
    }
    /*
     * The dirty bitmap is indexed in target pages, so a guest page smaller
     * than one cannot be asked about. Both product pathways are the other way
     * round (4 KiB x86 == target, 16 KiB arm64e > target), and answering for a
     * granularity the bitmap does not have would be a guess.
     */
    if (page_size < qemu_target_page_size()) {
        return 0;
    }

    pages = g_new(uint64_t, count);
    for (i = 0; i < count; i++) {
        pages[i] = gpas[i] & ~((uint64_t)page_size - 1);
    }
    qsort(pages, count, sizeof(*pages), reims_vgpu_dirty_cmp_u64);
    n = 0;
    for (i = 0; i < count; i++) {
        if (n == 0 || pages[n - 1] != pages[i]) {
            pages[n++] = pages[i];
        }
    }

    s = g_new0(ReimsVgpuDirtySet, 1);
    s->pages = pages;
    s->page_gen = g_new0(uint64_t, n);
    s->count = n;
    s->page_size = page_size;

    qemu_mutex_lock(&d->lock);
    /*
     * One harvest, and a second only if that harvest has to turn logging on.
     *
     * This used to be an unconditional two, because "the first harvest after
     * this call is the one that may still be turning logging on for these
     * pages, so writes older than it were never recorded". That reason is
     * right, and it is a reason about the *first* set to name a region — not
     * about every set. Guest RAM is one MemoryRegion, so once any surface has
     * been tracked, logging is already on for every later one, no harvest ever
     * enables anything, and the second harvest was waiting for something that
     * had already happened.
     *
     * It is not free to wait. While the set is unarmed its generation reads
     * back 0, `HostOps::guest_write_gen` maps that to "cannot tell", the
     * mapper-ref-texture LOAD elision refuses with `t11_gw_ref_no_stamp`, and
     * every draw onto that surface pays a whole-frame seed read out of guest
     * pages plus a
     * whole-frame staging upload. The window is counted in harvests and
     * harvests are driven by guest doorbells, so on a quiet desktop it lasts as
     * long as the guest stays quiet — which is exactly when a draw arriving
     * into it is a visible hitch (measured at 12-65 ms per draw, against
     * 0.2 ms driven).
     *
     * The second harvest is still taken when it is owed: `reims_vgpu_dirty_harvest`
     * pushes every set still waiting when it had to enable logging, so a set
     * created before the region was logged arms exactly as late as it used to.
     */
    s->arm_at = d->harvests + 1;
    d->next_token++;
    token = d->next_token;
    g_hash_table_insert(d->sets, g_memdup2(&token, sizeof(token)), s);
    qemu_mutex_unlock(&d->lock);
    return token;
}

void reims_vgpu_dirty_untrack(ReimsVgpuDirty *d, uint64_t token)
{
    if (!d || token == 0) {
        return;
    }
    qemu_mutex_lock(&d->lock);
    g_hash_table_remove(d->sets, &token);
    qemu_mutex_unlock(&d->lock);
}

uint64_t reims_vgpu_dirty_gen(ReimsVgpuDirty *d, uint64_t token)
{
    ReimsVgpuDirtySet *s;
    uint64_t gen = 0;

    if (!d || token == 0) {
        return 0;
    }
    qemu_mutex_lock(&d->lock);
    s = g_hash_table_lookup(d->sets, &token);
    if (s) {
        gen = s->gen;
    }
    d->reads_since_harvest++;
    qemu_mutex_unlock(&d->lock);
    return gen;
}

int64_t reims_vgpu_dirty_written_since(ReimsVgpuDirty *d, uint64_t token,
                                       uint64_t since_gen, uint64_t *out,
                                       size_t max)
{
    ReimsVgpuDirtySet *s;
    int64_t found = 0;
    size_t p;

    if (!d || token == 0 || out == NULL) {
        return -1;
    }
    qemu_mutex_lock(&d->lock);
    s = g_hash_table_lookup(d->sets, &token);
    /*
     * `since_gen == 0` is a caller that never recorded a readable observation,
     * and `s->gen == 0` is a set still inside its startup window. Neither can
     * be compared against a page stamp, and both mean the same thing to the
     * caller as an unknown token does.
     */
    if (!s || s->gen == 0 || since_gen == 0) {
        qemu_mutex_unlock(&d->lock);
        return -1;
    }
    for (p = 0; p < s->count; p++) {
        if (s->page_gen[p] <= since_gen) {
            continue;
        }
        if ((size_t)found == max) {
            /* Truncation would read as "these pages and no others". */
            qemu_mutex_unlock(&d->lock);
            return -1;
        }
        out[found++] = s->pages[p];
    }
    /*
     * Counted like reims_vgpu_dirty_gen(): a caller asking this question is a
     * consumer of the report, so the next harvest has something to tell it and
     * must not be skipped.
     */
    d->reads_since_harvest++;
    qemu_mutex_unlock(&d->lock);
    return found;
}

typedef struct ReimsVgpuDirtyRamScan {
    ReimsVgpuDirty *d;
    int n;
} ReimsVgpuDirtyRamScan;

static bool reims_vgpu_dirty_ram_range(Int128 start, Int128 len,
                                       const MemoryRegion *mr,
                                       hwaddr offset_in_region, void *opaque)
{
    ReimsVgpuDirtyRamScan *scan = opaque;
    ReimsVgpuDirty *d = scan->d;
    ReimsVgpuDirtySlice *sl;

    if (!mr->ram || !int128_nz(len)) {
        return false;
    }
    if (scan->n == d->slices_cap) {
        d->slices_cap = d->slices_cap ? d->slices_cap * 2 : 16;
        d->slices = g_renew(ReimsVgpuDirtySlice, d->slices, d->slices_cap);
    }
    sl = &d->slices[scan->n++];
    sl->logged = false;
    sl->needs_log = false;
    /*
     * Dropping const: the FlatView hands out a const view of a region this
     * harvest goes on to read a dirty bitmap for, turn logging on for, and
     * clear bits in, all of which are non-const operations on a region the
     * caller holds the BQL over.
     */
    sl->mr = (MemoryRegion *)mr;
    sl->gpa = int128_get64(start);
    sl->len = int128_get64(len);
    sl->offset = offset_in_region;
    sl->ram_addr = memory_region_get_ram_addr(sl->mr) + offset_in_region;
    return false;
}

/*
 * The guest-RAM ranges of the system address space. Returns how many were
 * recorded.
 *
 * This asks the FlatView rather than reconstructing it, and the difference is
 * not a simplification. It used to cut a monotone hull of every GPA ever
 * tracked into slices by walking it with address_space_translate(), and that
 * walk could not be made correct: address_space_translate_internal() clamps
 * `plen` to the section *only when the section is RAM* (system/physmem.c). The
 * first non-RAM byte in the hull therefore returned the whole remaining length,
 * and the walk recorded one "not RAM" slice covering every guest page above it.
 *
 * On x86 q35 low RAM ends below the PCI hole and high RAM begins at 4 GiB, so
 * one tracked page landing in low RAM while others sat above the hole put every
 * page in high RAM into that slice. The harvest reads a page it cannot resolve
 * as written — the conservative answer, and the right one for a page that is
 * genuinely gone — so every tracked surface then reported itself permanently
 * overwritten by the guest. The hull never shrank and the clear pass skipped
 * unresolved pages, so nothing could undo it for the life of the VM.
 *
 * Downstream that is a guest-visible latch, not a slow path: every host-side
 * copy of a surface is declared stale, the mapper-ref-texture sampled rung
 * refuses its resident and merges guest pages that never held the composite,
 * and deferred
 * render windows report `deferred_flush_clobber`. It reads as backdrops going
 * transparent and popover geometry breaking, until the guest is rebooted.
 *
 * A range is kept whenever it is `ram`, including device BARs and flash. A
 * tracked page that lands in one is a page this device must answer for rather
 * than skip, and nothing turns logging on for a region until a tracked page
 * resolves into it.
 *
 * Every such range is recorded; the table grows to fit rather than stopping at
 * a bound. A bound here is not a budget that costs latency when it binds — a
 * dropped range makes every page in it read as written for the life of the VM,
 * which is the same latch by a smaller door, and one no counter would name.
 * Growth is amortized and the steady state reallocates nothing, because the
 * table is owned across harvests and the flat view's shape does not churn.
 */
static int reims_vgpu_dirty_ram_slices(ReimsVgpuDirty *d)
{
    ReimsVgpuDirtyRamScan scan = { d, 0 };

    RCU_READ_LOCK_GUARD();
    flatview_for_each_range(address_space_to_flatview(&address_space_memory),
                            reims_vgpu_dirty_ram_range, &scan);
    return scan.n;
}

/* Which slice holds `gpa`, or -1. Slices are few and ordered, so a linear
 * scan beats any structure these counts would justify. */
static int reims_vgpu_dirty_slice_of(const ReimsVgpuDirtySlice *slices, int n,
                                     uint64_t gpa)
{
    int i;

    for (i = 0; i < n; i++) {
        if (gpa >= slices[i].gpa && gpa - slices[i].gpa < slices[i].len) {
            return i;
        }
    }
    return -1;
}

/*
 * Was any target page of the guest page at `gpa` written?
 *
 * A pure read of the bitmap, deliberately: the answer must be the same for
 * every set holding the page, so nothing is consumed here. Clearing happens
 * once, after every set has been evaluated.
 */
static bool reims_vgpu_dirty_page_written(const ReimsVgpuDirtySlice *sl,
                                          uint64_t gpa, uint64_t page_size)
{
    size_t target = qemu_target_page_size();
    ram_addr_t base = sl->ram_addr + (gpa - sl->gpa);
    uint64_t off;

    for (off = 0; off < page_size; off += target) {
        if (physical_memory_get_dirty_flag(base + off, DIRTY_MEMORY_VGA)) {
            return true;
        }
    }
    return false;
}

/*
 * Is the DIRTY_MEMORY_VGA bitmap actually reporting guest CPU writes on this
 * host?  Every verdict this device gives about a surface rests on it, and a
 * bitmap that never sets a bit is indistinguishable, from inside, from a guest
 * that never wrote.  ORCHARD_DIRTY_DEBUG=1 prints the split: harvests run,
 * pages inspected, and the ones a bitmap bit -- not a missing slice, not an
 * unlogged region -- reported written.
 */
static uint64_t dbg_harvests, dbg_pages, dbg_bitmap_hits, dbg_unlogged, dbg_noslice;

static void reims_vgpu_dirty_debug_report(void)
{
    static uint64_t last;
    const char *on = getenv("ORCHARD_DIRTY_DEBUG");

    if (!on || *on != '1' || dbg_harvests == last) {
        return;
    }
    if (dbg_harvests - last < 64) {
        return;
    }
    last = dbg_harvests;
    fprintf(stderr,
            "reims-dirty: harvests=%" PRIu64 " pages=%" PRIu64
            " bitmap_written=%" PRIu64 " unlogged=%" PRIu64
            " no_slice=%" PRIu64 "\n",
            dbg_harvests, dbg_pages, dbg_bitmap_hits, dbg_unlogged,
            dbg_noslice);
}

void reims_vgpu_dirty_harvest(ReimsVgpuDirty *d)
{
    ReimsVgpuDirtySlice *slices;
    g_autoptr(GArray) written = NULL;
    g_autoptr(GArray) hit = NULL;
    GHashTableIter it;
    gpointer key, val;
    int n, i;
    guint w;

    if (!d) {
        return;
    }

    qemu_mutex_lock(&d->lock);
    if (g_hash_table_size(d->sets) == 0 || d->reads_since_harvest == 0) {
        qemu_mutex_unlock(&d->lock);
        return;
    }
    qemu_mutex_unlock(&d->lock);

    n = reims_vgpu_dirty_ram_slices(d);
    slices = d->slices;
    /*
     * Whether this device was already recording writes to each range when the
     * sync below ran. Answered once per range rather than per page: the page
     * loop runs over every page of every tracked set, and a hash lookup there
     * is the harvest's inner loop.
     *
     * A range that is not logged yet is one whose bits mean nothing, so its
     * pages read as written below and it is queued for logging afterwards.
     * Enabling is deferred past the clear pass because memory_region_set_log()
     * is a MemoryRegion transaction that rebuilds the flat view, which would
     * leave this slice table describing a view that no longer exists.
     */
    for (i = 0; i < n; i++) {
        slices[i].logged = g_hash_table_contains(d->logged, slices[i].mr);
    }

    /* One sync for every logged region, then only reads. */
    memory_global_dirty_log_sync(false);

    written = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyWritten));
    /*
     * Indices of the current set's written pages. Held across the generation
     * update because the value to stamp them with is the generation this
     * harvest produces, which is not known until every page has been read.
     * Reused between sets so the harvest allocates once, not once per mapping.
     */
    hit = g_array_new(FALSE, FALSE, sizeof(size_t));
    qemu_mutex_lock(&d->lock);
    d->harvests++;
    d->reads_since_harvest = 0;
    dbg_harvests++;
    g_hash_table_iter_init(&it, d->sets);
    while (g_hash_table_iter_next(&it, &key, &val)) {
        ReimsVgpuDirtySet *s = val;
        bool any = false;
        bool unlogged = false;
        size_t p;

        g_array_set_size(hit, 0);
        for (p = 0; p < s->count; p++) {
            uint64_t gpa = s->pages[p];
            int si = reims_vgpu_dirty_slice_of(slices, n, gpa);

            dbg_pages++;

            /*
             * Not guest RAM this device can read a bitmap for, or RAM whose
             * writes this device was not yet recording when the sync ran.
             * Both mean the host cannot say the page is unwritten, and that
             * reads as written — per page as well as for the set, so a
             * per-page reader is no less conservative than the generation is.
             */
            if (si < 0) {
                dbg_noslice++;
                any = true;
                g_array_append_val(hit, p);
                continue;
            }
            if (!slices[si].logged) {
                dbg_unlogged++;
                slices[si].needs_log = true;
                unlogged = true;
                any = true;
                g_array_append_val(hit, p);
                continue;
            }
            if (reims_vgpu_dirty_page_written(&slices[si], gpa, s->page_size)) {
                ReimsVgpuDirtyWritten rec = { gpa, s->page_size };

                dbg_bitmap_hits++;
                any = true;
                g_array_append_val(hit, p);
                /* Recorded for the clear pass, which must not run until every
                 * set has read the bit — otherwise the first set harvested
                 * would consume the report of every set sharing the page. */
                g_array_append_val(written, rec);
            }
        }
        /*
         * Some page of this set sits in a region whose writes this device was
         * not yet recording when the sync above ran, so writes older than it
         * were never recorded and their absence is not evidence. Wait for a
         * harvest that covered every page.
         *
         * This is what lets `reims_vgpu_dirty_track` arm at +1 instead of +2.
         * The +2 was paying for this case on every set; this pays for it on the
         * sets it applies to, which after the first surface of a boot is none.
         * The test is `s->gen == 0` — never armed — because an already armed set
         * must not be sent back through its startup window: its pages read as
         * written above, which is the conservative answer and needs no help.
         */
        if (s->gen == 0 && unlogged) {
            s->arm_at = d->harvests + 1;
        }
        if (d->harvests < s->arm_at) {
            /* Startup window: an absence of reports says nothing about the
             * guest yet, so the generation stays unreadable. */
            s->gen = 0;
        } else if (s->gen == 0) {
            s->gen = 1;
        } else if (any) {
            s->gen++;
        }
        /*
         * Stamp after the generation, and only once it is readable. A page
         * stamped during the startup window would carry a generation no reader
         * can have recorded, and would then read as written forever.
         */
        if (s->gen != 0) {
            guint h;

            for (h = 0; h < hit->len; h++) {
                s->page_gen[g_array_index(hit, size_t, h)] = s->gen;
            }
        }
    }
    qemu_mutex_unlock(&d->lock);

    /*
     * Clear only the pages that came back written. Clearing re-protects, so
     * clearing the whole tracked window would make every page the guest has
     * mapped refault after every harvest; clearing what was written costs
     * exactly one refault per page the guest writes again, which is the work
     * the report is worth.
     *
     * Exactly-adjacent pages merge into one call because each call is a
     * hypervisor round trip. Nothing is widened: a merged range covers only
     * pages that were themselves written.
     */
    g_array_sort(written, reims_vgpu_dirty_cmp_written);
    w = 0;
    while (w < written->len) {
        ReimsVgpuDirtyWritten first = g_array_index(written, ReimsVgpuDirtyWritten, w);
        int si = reims_vgpu_dirty_slice_of(slices, n, first.gpa);
        uint64_t end = first.gpa + first.page_size;

        while (w + 1 < written->len) {
            ReimsVgpuDirtyWritten next =
                g_array_index(written, ReimsVgpuDirtyWritten, w + 1);

            if (next.gpa + next.page_size <= end) {
                /* Same page recorded by a second set, or a smaller page
                 * already inside the run. */
                w++;
                continue;
            }
            if (next.gpa != end ||
                reims_vgpu_dirty_slice_of(slices, n, next.gpa) != si) {
                break;
            }
            end = next.gpa + next.page_size;
            w++;
        }
        if (si >= 0) {
            memory_region_reset_dirty(
                slices[si].mr, slices[si].offset + (first.gpa - slices[si].gpa),
                end - first.gpa, DIRTY_MEMORY_VGA);
        }
        w++;
    }

    /*
     * Start recording writes to every region a tracked page resolved into and
     * that was not being recorded for the sync above. Last, because this is a
     * MemoryRegion transaction: it rebuilds the flat view, and the slice table
     * every pass above it describes the old one.
     *
     * Nothing is missed by the delay. Every page in such a region already read
     * as written — which is what a region with no recorded history has to say —
     * and no set whose pages were among them armed on this harvest.
     */
    for (i = 0; i < n; i++) {
        if (!slices[i].needs_log) {
            continue;
        }
        /*
         * One MemoryRegion can supply several ranges — on x86 q35 guest RAM is
         * one region aliased below and above the PCI hole — so the insert is
         * what decides, not the flag. g_hash_table_add() reports whether the
         * key was new, which keeps the reference this table owns at one per
         * region however many of its ranges asked.
         */
        if (g_hash_table_add(d->logged, slices[i].mr)) {
            memory_region_ref(slices[i].mr);
            memory_region_set_log(slices[i].mr, true, DIRTY_MEMORY_VGA);
        }
    }

    reims_vgpu_dirty_debug_report();
}
