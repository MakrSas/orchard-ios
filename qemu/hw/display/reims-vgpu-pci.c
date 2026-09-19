/*
 * Thin PCI QEMU shim for the product paravirtualized GPU device.
 *
 * Type name: reims-vgpu-pci (PCI IDs vendor 0x106B / device 0xEEEE).
 * Sibling of reims-vgpu-mmio (sysbus/vmapple): same HostOps + Rust staticlib
 * boundary. C owns only PCI/QOM/BAR/MSI/console/BH — no protocol or GPU logic
 * Product behaviour lives in crates/reims-vgpu.
 *
 * Topology: -vga none + pcie-root-port (non-zero bus) + this device;
 * class VGA 0x030000; MSI vector 0; BAR0 16 KiB (control +0x1000).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "qemu/main-loop.h"
#include "qemu/aio.h"
#include "qemu/thread.h"
#include "qapi/error.h"
#include "hw/pci/pci_device.h"
#include "hw/pci/msi.h"
#include "hw/pci/pci.h"
#include "hw/pci/pcie.h"
#include "qom/object.h"
#include "system/address-spaces.h"
#include "system/memory.h"
#include "system/ramblock.h"
#include "system/runstate.h"
#include "ui/console.h"
#include "ui/surface.h"
#include "trace.h"
#include "reims_vgpu_qemu_abi.h"
#include "reims-vgpu-dirty.h"
#include "reims-vgpu-shim.h"

#define TYPE_REIMS_VGPU_PCI "reims-vgpu-pci"
OBJECT_DECLARE_SIMPLE_TYPE(ReimsVGPUPCIState, REIMS_VGPU_PCI)

typedef struct ReimsVGPUPCIPageView {
    void *ptr;
    size_t len;
} ReimsVGPUPCIPageView;

/*
 * BAR0 = full gfx window; control block at +0x1000. The size is the shared
 * REIMS_VGPU_GFX_MMIO_SIZE — Rust bounds its register store against the same
 * number, and a private copy here is a window the guest can address past it.
 */
/* BAR1 = linear UEFI GOP framebuffer (BGRA8 at the EFI boot mode + headroom). */
#define REIMS_VGPU_PCI_FB_SIZE   (16u * 1024u * 1024u)
/*
 * These two IDs are what the guest's paravirt GPU driver matches on, and the
 * match decides which Metal plugin the guest loads. That has a consequence on
 * macOS 26 that is worth stating here, because it is invisible from this file
 * and it is not a defect anything on this side can repair.
 *
 * The guest ships two personalities for one paravirt GPU:
 *
 *   - one matching an IOPCIDevice on exactly these IDs, whose Metal plugin
 *     descends from Metal's IOAccelerator device base class;
 *   - one matching a *device-tree* node by name, on an ARM platform provider,
 *     whose Metal plugin descends from the IOGPUFamily device base class.
 *
 * Only the second plugin implements the device-architecture property that
 * Metal's binary-archive loader asserts on. On macOS 26 the icon-rendering
 * path loads a precompiled binary archive, so on this pathway that assert
 * fires and the icon renderers abort — repeatedly, since each icon re-render
 * retries. The other personality cannot be reached from here at any PCI
 * identity: it requires an ARM platform provider that does not exist on an
 * x86_64 guest, and its kext declares a dependency on the ARM platform driver.
 *
 * So this is structural to the PCI pathway rather than a value to tune. Do not
 * spend a boot probing device-info keys for it — the capability table is not in
 * that code path at all. Changing these IDs does not reach the other
 * personality; it only stops the driver matching us.
 */
#define REIMS_VGPU_PCI_VENDOR    0x106B
#define REIMS_VGPU_PCI_DEVICE    0xEEEE
/* BAR1's own pixel size, not a shared constant: it describes this shim's GOP
 * layout, which Rust reaches only through efi_console_copy's explicit stride. */
#define REIMS_VGPU_PCI_EFI_BPP   4u

struct ReimsVGPUPCIState {
    PCIDevice parent_obj;

    MemoryRegion iomem_gfx;
    /*
     * BAR1 linear FB (same PCI device — not a second display). UEFI GOP option
     * ROM + OpenCore paint here until the product present path latches.
     */
    MemoryRegion fb_vram;
    QemuConsole *con;
    DisplaySurface *surface;
    bool new_frame_ready;

    ReimsVgpuHostOps host_ops;
    /* Hypervisor dirty-bitmap adapter: the only witness for a write to a
     * surface's guest pages that no device operation made. */
    ReimsVgpuDirty *dirty;
    uint64_t rust_handle;
    /* Caller-owned packed aliases over file-backed guest pages. Direct
     * RAMBlock pointers never enter this list; constructed views leave through
     * unmap_pages when the Rust mapping lifetime ends, with device exit as the
     * final safety net. */
    GArray *page_views;
    uint64_t page_view_live_bytes;
    uint64_t page_view_live_pages;
    uint64_t page_views_created;
    uint64_t page_views_destroyed;

    /* Ordered Rust FIFO/render drain owner. The AIO BH only applies completed
     * HostActions; it never translates shaders or waits for GPU work. */
    QemuThread drain_thread;
    QemuMutex drain_mutex;
    QemuCond drain_cond;
    QEMUBH *action_bh;
    /*
     * Steady vblank + Dekker-rescue heartbeat. Keep its wait on a dedicated
     * thread: main-loop timers are delayed by input/QMP/display work during
     * window drags, which starves the guest display time base.
     */
    QemuThread heartbeat_thread;
    QemuMutex heartbeat_mutex;
    QemuCond heartbeat_cond;
    bool drain_pending;
    bool drain_stopping;
    bool drain_started;
    bool heartbeat_stopping;
    bool heartbeat_started;
    Notifier shutdown_notifier;
    bool shutdown_notifier_registered;
};

/* ---------- HostOps (GPA R/W + BH; Linux has no map_pages / xreg) ---------- */

static int reims_vgpu_pci_read_xreg(void *ctx, uint32_t index, uint64_t *out)
{
    (void)ctx;
    (void)index;
    (void)out;
    /* x86 product path: no arm xreg handoff. */
    return -1;
}

static int reims_vgpu_pci_map_pages(void *ctx, const uint64_t *gpas, size_t count,
                             void **out_ptr, ReimsVgpuMapPagesFailure *failure)
{
    ReimsVGPUPCIState *s = ctx;
    /* x86 guest page granularity (host-pointer import view stride). */
    const hwaddr page = (hwaddr)1 << REIMS_VGPU_GUEST_PAGE_SHIFT_X86_64;
    uint8_t *base = NULL;
    MemoryRegion *base_mr = NULL;
    size_t i;

    if (failure) {
        *failure = (ReimsVgpuMapPagesFailure) {
            .stage = REIMS_VGPU_MAP_PAGES_FAILURE_NONE,
        };
    }
    if (!s || !gpas || count == 0 || !out_ptr || !failure ||
        count > SIZE_MAX / page) {
        if (failure) {
            failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_INVALID_GUEST_PAGE;
            failure->page_index = UINT64_MAX;
        }
        return -1;
    }

    /*
     * Packed contiguous host-VA view: guest RAM is already mmap'd in the QEMU
     * process. Callers (Rust map_pages consumers) treat page i as
     * `base + i * page` — so every GPA in the list must alias host VA at that
     * packed offset (same RAMBlock, sequential host pages). Non-sequential /
     * cross-MR / non-RAM lists return -1; callers multi-import maximal runs
     * or fail closed (`not_contig`).
     *
     * Note: do **not** accept `hva == base + (gpas[i] - gpas[0])` for sparse
     * GPAs — that would return success while page i is not at base+i*page,
     * silently mis-aliasing multi-page views.
     */
    rcu_read_lock();

    /*
     * Maximal GPA runs are the common large-surface path.  One address-space
     * translation covering the complete sequential run proves the same
     * RAMBlock/HVA contract as the page loop below.  Avoiding thousands of
     * identical translations is important for full-frame staging/writeback;
     * retain the page loop for aliases and region-boundary cases.
     */
    if (count <= UINT64_MAX / page &&
        gpas[0] <= UINT64_MAX - (count - 1) * page) {
        hwaddr total = count * page;
        bool sequential = true;

        for (i = 1; i < count; i++) {
            if (gpas[i] != gpas[0] + i * page) {
                sequential = false;
                break;
            }
        }
        if (sequential) {
            hwaddr xlat, plen = total;
            MemoryRegion *mr;
            uint8_t *hva;

            mr = address_space_translate(&address_space_memory, gpas[0],
                                         &xlat, &plen, true,
                                         MEMTXATTRS_UNSPECIFIED);
            if (mr && memory_region_is_ram(mr) && plen >= total) {
                hva = (uint8_t *)memory_region_get_ram_ptr(mr) + xlat;
                if (((uintptr_t)hva & (page - 1)) == 0) {
                    rcu_read_unlock();
                    *out_ptr = hva;
                    return 0;
                }
            }
        }
    }

    for (i = 0; i < count; i++) {
        hwaddr xlat, plen = page;
        MemoryRegion *mr;
        uint8_t *hva;

        mr = address_space_translate(&address_space_memory, gpas[i], &xlat,
                                     &plen, true, MEMTXATTRS_UNSPECIFIED);
        if (!mr || !memory_region_is_ram(mr) || plen < page) {
            goto fail;
        }
        hva = (uint8_t *)memory_region_get_ram_ptr(mr) + xlat;
        if (i == 0) {
            base = hva;
            base_mr = mr;
            if (((uintptr_t)base & (page - 1)) != 0) {
                goto fail; /* base must be page-aligned for import */
            }
        } else if (mr != base_mr || hva != base + i * page) {
            goto fail; /* page i not packed-contig with the run */
        }
    }
    rcu_read_unlock();

    *out_ptr = base;
    return 0;

fail:
    rcu_read_unlock();

#if defined(_WIN32)
    /*
     * No POSIX mmap on Windows: packed aliasing views are unavailable, so every
     * scattered GVA mapping takes the product copying rails instead.
     */
    return -1;
#else
    /*
     * A task GVA is linear even when its guest-physical pages are not. When
     * guest RAM is shared file-backed memory, recreate that virtual shape with
     * one MAP_SHARED view. This is an alias, not a copy: guest CPU writes and
     * GPU reads still address the same pages.
     *
     * Reserve the complete range first, then replace it page by page. A failed
     * page tears down the whole reservation, so callers never receive a
     * partially packed view.
     */
    {
        size_t total = count * page;
        uint8_t *view = mmap(NULL, total, PROT_NONE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ReimsVGPUPCIPageView held;

        if (view == MAP_FAILED) {
            failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_RESERVATION;
            failure->host_errno = errno;
            return -1;
        }
        rcu_read_lock();
        for (i = 0; i < count; i++) {
            hwaddr xlat, plen = page;
            MemoryRegion *mr;
            uint8_t *hva;
            RAMBlock *rb;
            ram_addr_t rb_offset;
            ram_addr_t fd_offset;
            int fd;
            void *mapped;

            mr = address_space_translate(&address_space_memory, gpas[i],
                                         &xlat, &plen, true,
                                         MEMTXATTRS_UNSPECIFIED);
            if (!mr || !memory_region_is_ram(mr) || plen < page) {
                failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_INVALID_GUEST_PAGE;
                failure->page_index = i;
                goto alias_fail_locked;
            }
            hva = (uint8_t *)memory_region_get_ram_ptr(mr) + xlat;
            rb = qemu_ram_block_from_host(hva, false, &rb_offset);
            if (!rb || !qemu_ram_is_shared(rb)) {
                failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_ALIAS;
                failure->page_index = i;
                goto alias_fail_locked;
            }
            fd = qemu_ram_get_fd(rb);
            if (fd < 0 || rb_offset > RAM_ADDR_MAX - qemu_ram_get_fd_offset(rb)) {
                failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_ALIAS;
                failure->page_index = i;
                goto alias_fail_locked;
            }
            fd_offset = qemu_ram_get_fd_offset(rb) + rb_offset;
            if ((fd_offset & (page - 1)) != 0) {
                failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_ALIAS;
                failure->page_index = i;
                goto alias_fail_locked;
            }
            mapped = mmap(view + i * page, page, PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_FIXED, fd, fd_offset);
            if (mapped == MAP_FAILED) {
                failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_ALIAS;
                failure->host_errno = errno;
                failure->page_index = i;
                goto alias_fail_locked;
            }
        }
        rcu_read_unlock();

        held.ptr = view;
        held.len = total;
        g_array_append_val(s->page_views, held);
        s->page_view_live_bytes += total;
        s->page_view_live_pages += count;
        s->page_views_created++;
        *out_ptr = view;
        return 0;

alias_fail_locked:
        rcu_read_unlock();
        munmap(view, total);
        return -1;
    }
#endif
}

static void reims_vgpu_pci_unmap_pages(void *ctx, void *ptr, size_t len)
{
    ReimsVGPUPCIState *s = ctx;
    size_t i;

    if (!s || !s->page_views || !ptr || len == 0) {
        return;
    }

    for (i = 0; i < s->page_views->len; i++) {
        ReimsVGPUPCIPageView *view =
            &g_array_index(s->page_views, ReimsVGPUPCIPageView, i);

        if (view->ptr != ptr || view->len != len) {
            continue;
        }
        munmap(view->ptr, view->len);
        s->page_view_live_bytes -= view->len;
        s->page_view_live_pages -= view->len >> REIMS_VGPU_GUEST_PAGE_SHIFT_X86_64;
        s->page_views_destroyed++;
        g_array_remove_index_fast(s->page_views, i);
        return;
    }
    /* A direct RAMBlock pointer is borrowed from QEMU and has no entry. */
}

static int reims_vgpu_pci_page_alias_census(
    void *ctx, ReimsVgpuPageAliasCensus *out)
{
    ReimsVGPUPCIState *s = ctx;

    if (!s || !s->page_views || !out) {
        return -1;
    }
    *out = (ReimsVgpuPageAliasCensus) {
        .live = s->page_views->len,
        .live_bytes = s->page_view_live_bytes,
        .live_pages = s->page_view_live_pages,
        .created = s->page_views_created,
        .destroyed = s->page_views_destroyed,
    };
    return 0;
}

/*
 * Guest-write tracking. A surface's storage is plain guest RAM: the guest CPU
 * stores into it with no device operation, so nothing the Rust device counts
 * can witness such a store and every host-side copy of those pages is stale
 * from that instant. These three forward to the shared dirty-bitmap adapter.
 */
static uint64_t reims_vgpu_pci_track_guest_writes(void *ctx, const uint64_t *gpas,
                                                  size_t count, size_t page_size)
{
    ReimsVGPUPCIState *s = ctx;

    return reims_vgpu_dirty_track(s->dirty, gpas, count, page_size);
}

static void reims_vgpu_pci_untrack_guest_writes(void *ctx, uint64_t token)
{
    ReimsVGPUPCIState *s = ctx;

    reims_vgpu_dirty_untrack(s->dirty, token);
}

static uint64_t reims_vgpu_pci_guest_write_gen(void *ctx, uint64_t token)
{
    ReimsVGPUPCIState *s = ctx;

    return reims_vgpu_dirty_gen(s->dirty, token);
}

static int64_t reims_vgpu_pci_guest_written_pages(void *ctx, uint64_t token,
                                         uint64_t since_gen, uint64_t *out,
                                         size_t max)
{
    ReimsVGPUPCIState *s = ctx;

    return reims_vgpu_dirty_written_since(s->dirty, token, since_gen, out, max);
}

static void reims_vgpu_pci_bh(void *opaque);
static void reims_vgpu_pci_deliver_actions(ReimsVGPUPCIState *s);
static void reims_vgpu_pci_apply_action(ReimsVGPUPCIState *s, const ReimsVgpuHostAction *a);

static void reims_vgpu_pci_schedule_bh(void *ctx)
{
    ReimsVGPUPCIState *s = ctx;

    qemu_mutex_lock(&s->drain_mutex);
    s->drain_pending = true;
    qemu_cond_signal(&s->drain_cond);
    qemu_mutex_unlock(&s->drain_mutex);
}

/*
 * Prompt HostAction delivery (IRQ pulses / cursor moves): schedule the
 * action BH from any thread so acks reach the guest while the drain worker
 * is still executing a tranche. qemu_bh_schedule is thread-safe.
 */
static void reims_vgpu_pci_notify_actions(void *ctx)
{
    ReimsVGPUPCIState *s = ctx;

    if (s->action_bh) {
        qemu_bh_schedule(s->action_bh);
    }
}

static void reims_vgpu_pci_deliver_actions(ReimsVGPUPCIState *s)
{
    ReimsVgpuHostAction action;
    int rc;

    if (s->rust_handle == 0) {
        return;
    }
    while ((rc = reims_vgpu_qemu_device_pop_action(s->rust_handle, &action)) ==
           REIMS_VGPU_QEMU_OK) {
        reims_vgpu_pci_apply_action(s, &action);
    }
}

/* ---------- Console ---------- */

static void reims_vgpu_pci_set_mode(ReimsVGPUPCIState *s, uint32_t width,
                             uint32_t height)
{
    if (width == 0 || height == 0 ||
        width > REIMS_VGPU_MAX_SCANOUT_DIM || height > REIMS_VGPU_MAX_SCANOUT_DIM) {
        return;
    }
    if (s->surface &&
        surface_width(s->surface) == width &&
        surface_height(s->surface) == height) {
        return;
    }

    s->surface = qemu_create_displaysurface(width, height);
    if (s->con) {
        qemu_console_set_surface(s->con, s->surface);
    }
    trace_reims_vgpu_pci_mode_change(width, height);
}

/*
 * Copy BAR1 linear BGRA8 into the QEMU DisplaySurface (UEFI GOP / OpenCore).
 * Used when the Metal/product present path has not yet latched a front buffer.
 */
static bool reims_vgpu_pci_copy_gop_fb(ReimsVGPUPCIState *s)
{
    uint8_t *src;
    uint8_t *dst;
    uint32_t dst_stride;
    uint32_t src_stride;
    uint32_t w = REIMS_VGPU_EFI_BOOT_WIDTH;
    uint32_t h = REIMS_VGPU_EFI_BOOT_HEIGHT;
    uint32_t y;

    if (!s->surface || !memory_region_is_ram(&s->fb_vram)) {
        return false;
    }
    src = memory_region_get_ram_ptr(&s->fb_vram);
    if (!src) {
        return false;
    }
    if (surface_width(s->surface) != w || surface_height(s->surface) != h) {
        reims_vgpu_pci_set_mode(s, w, h);
        if (!s->surface) {
            return false;
        }
    }
    dst = surface_data(s->surface);
    dst_stride = surface_stride(s->surface);
    src_stride = w * REIMS_VGPU_PCI_EFI_BPP;
    if (dst_stride == src_stride) {
        memcpy(dst, src, (size_t)src_stride * h);
    } else {
        for (y = 0; y < h; y++) {
            memcpy(dst + (size_t)y * dst_stride,
                   src + (size_t)y * src_stride, src_stride);
        }
    }
    return true;
}

/*
 * Pre-boundary early console: prefer guest-programmed EFI FB (MMIO 0x1210)
 * when the kernel relocates the video console off BAR1 into guest RAM
 * (serial: "console relocated to 0xf1000000"). Fall back to BAR1 GOP.
 * Contract: efi_fb_start / stride from Apple EFI block — not serial scrape.
 */
static bool reims_vgpu_pci_copy_early_console(ReimsVGPUPCIState *s)
{
    uint8_t *dst;
    uint32_t dst_stride;
    uint32_t w = REIMS_VGPU_EFI_BOOT_WIDTH;
    uint32_t h = REIMS_VGPU_EFI_BOOT_HEIGHT;
    int rc;

    if (!s->surface) {
        reims_vgpu_pci_set_mode(s, w, h);
        if (!s->surface) {
            return false;
        }
    }
    if (surface_width(s->surface) != w || surface_height(s->surface) != h) {
        reims_vgpu_pci_set_mode(s, w, h);
        if (!s->surface) {
            return false;
        }
    }
    if (s->rust_handle != 0) {
        dst = surface_data(s->surface);
        dst_stride = surface_stride(s->surface);
        rc = reims_vgpu_qemu_efi_console_copy(s->rust_handle, dst, dst_stride, w, h);
        if (rc == REIMS_VGPU_QEMU_OK) {
            return true;
        }
    }
    return reims_vgpu_pci_copy_gop_fb(s);
}

/*
 * Paint a named guest mapping into the QEMU surface (pre- or post-boundary).
 * Pre-boundary early writebacks and post-boundary DisplaySwap both use this
 * path. Return true when the surface was updated.
 */
static bool reims_vgpu_pci_paint_scanout(ReimsVGPUPCIState *s, uint32_t mapping_id,
                                  uint32_t width, uint32_t height,
                                  uint32_t generation)
{
    uint8_t *dst;
    uint32_t stride;
    int rc;

    if (!s->surface || s->rust_handle == 0) {
        return false;
    }
    if (width == 0 || height == 0) {
        return false;
    }
    if (surface_width(s->surface) != width ||
        surface_height(s->surface) != height) {
        reims_vgpu_pci_set_mode(s, width, height);
        if (!s->surface) {
            return false;
        }
    }
    dst = surface_data(s->surface);
    stride = surface_stride(s->surface);
    rc = reims_vgpu_qemu_scanout_copy(s->rust_handle, mapping_id, dst, stride,
                               width, height, generation);
    if (rc == REIMS_VGPU_QEMU_OK) {
        s->new_frame_ready = true;
        trace_reims_vgpu_pci_scanout(mapping_id, width, height);
        /* Push retain immediately — waiting only for the next gfx_update can
         * leave a black surface if a later EMPTY/Unchanged path races. */
        if (s->con) {
            qemu_console_update_full(s->con);
            s->new_frame_ready = false;
        }
        return true;
    }
    return false;
}

static void reims_vgpu_pci_apply_scanout(ReimsVGPUPCIState *s, uint32_t mapping_id,
                                  uint32_t width, uint32_t height,
                                  uint32_t generation)
{
    if (!reims_vgpu_shim_scanout_may_paint(s->rust_handle, mapping_id)) {
        return;
    }
    (void)reims_vgpu_pci_paint_scanout(s, mapping_id, width, height, generation);
}

/*
 * Whether the host-owned window (kb host-window) is requested via REIMS_VGPU_WINDOW.
 * Presence enables it; 0/off/no/false explicitly disable so REIMS_VGPU_WINDOW=0 opts
 * out (vm/boot-x86.sh defaults it on for this device, unset to disable). Env
 * plumbing only — the window itself lives entirely in the staticlib.
 */
static bool reims_vgpu_pci_window_requested(void)
{
    const char *v = getenv("REIMS_VGPU_WINDOW");

    if (!v || v[0] == '\0') {
        return false;
    }
    if (!strcmp(v, "0") || !strcasecmp(v, "off") || !strcasecmp(v, "no") ||
        !strcasecmp(v, "false")) {
        return false;
    }
    return true;
}

static void reims_vgpu_pci_apply_action(ReimsVGPUPCIState *s, const ReimsVgpuHostAction *a)
{
    PCIDevice *pdev = PCI_DEVICE(s);

    switch (a->kind) {
    case REIMS_VGPU_HOST_ACTION_IRQ_GFX:
        if (msi_enabled(pdev)) {
            msi_notify(pdev, 0);
        }
        trace_reims_vgpu_pci_irq_gfx();
        break;
    case REIMS_VGPU_HOST_ACTION_IRQ_IOSFC:
        /* Single MSI vector; pulse same vector (guest demuxes by status). */
        if (msi_enabled(pdev)) {
            msi_notify(pdev, 0);
        }
        break;
    case REIMS_VGPU_HOST_ACTION_SCANOUT:
        reims_vgpu_pci_apply_scanout(s, (uint32_t)a->a0, (uint32_t)a->a1,
                              (uint32_t)a->a2, (uint32_t)a->a3);
        break;
    case REIMS_VGPU_HOST_ACTION_CURSOR:
        if (s->con) {
            qemu_console_set_mouse(s->con, (int)a->a0, (int)a->a1, a->a2 != 0);
        }
        break;
    case REIMS_VGPU_HOST_ACTION_INPUT_KEY:
        reims_vgpu_shim_input_key(s->con, (uint32_t)a->a0, a->a1 != 0);
        break;
    case REIMS_VGPU_HOST_ACTION_INPUT_POINTER_MOVE:
        reims_vgpu_shim_input_pointer_move(s->con, (uint32_t)a->a0, (uint32_t)a->a1,
                                   (uint32_t)a->a2, (uint32_t)a->a3);
        break;
    case REIMS_VGPU_HOST_ACTION_INPUT_POINTER_BUTTON:
        reims_vgpu_shim_input_button(s->con, (uint32_t)a->a0, a->a1 != 0);
        break;
    case REIMS_VGPU_HOST_ACTION_WINDOW_CLOSED:
        /* The host window is the VM's display; closing it shuts the VM down. */
        qemu_system_shutdown_request(SHUTDOWN_CAUSE_HOST_UI);
        break;
    case REIMS_VGPU_HOST_ACTION_CURSOR_GLYPH: {
        ReimsVgpuCursorGlyphInfo info;
        g_autofree uint32_t *pixels = NULL;
        QEMUCursor *c;
        int rc;

        if (!s->con || s->rust_handle == 0) {
            break;
        }
        rc = reims_vgpu_qemu_cursor_glyph_info(s->rust_handle, &info);
        if (rc != REIMS_VGPU_QEMU_OK || info.width == 0 || info.height == 0 ||
            info.pixel_count == 0 ||
            info.pixel_count != info.width * info.height) {
            break;
        }
        pixels = g_new(uint32_t, info.pixel_count);
        rc = reims_vgpu_qemu_cursor_glyph_copy(s->rust_handle, pixels,
                                        info.pixel_count);
        if (rc != REIMS_VGPU_QEMU_OK) {
            break;
        }
        c = cursor_alloc(info.width, info.height);
        if (!c) {
            break;
        }
        c->hot_x = info.hot_x;
        c->hot_y = info.hot_y;
        memcpy(c->data, pixels, (size_t)info.pixel_count * sizeof(uint32_t));
        qemu_console_set_cursor(s->con, c);
        cursor_unref(c);
        break;
    }
    case REIMS_VGPU_HOST_ACTION_TRACE:
    case REIMS_VGPU_HOST_ACTION_NONE:
    default:
        break;
    }
}

static void reims_vgpu_pci_bh(void *opaque)
{
    ReimsVGPUPCIState *s = opaque;

    if (s->rust_handle == 0) {
        return;
    }
    reims_vgpu_pci_deliver_actions(s);
}

static void *reims_vgpu_pci_drain_thread(void *opaque)
{
    ReimsVGPUPCIState *s = opaque;

    for (;;) {
        int rc;

        qemu_mutex_lock(&s->drain_mutex);
        while (!s->drain_pending && !s->drain_stopping) {
            qemu_cond_wait(&s->drain_cond, &s->drain_mutex);
        }
        if (s->drain_stopping) {
            qemu_mutex_unlock(&s->drain_mutex);
            break;
        }
        s->drain_pending = false;
        qemu_mutex_unlock(&s->drain_mutex);

        rc = reims_vgpu_qemu_device_drain(s->rust_handle);
        if (rc != REIMS_VGPU_QEMU_OK) {
            qemu_log_mask(LOG_GUEST_ERROR, "%s: worker drain failed rc=%d\n",
                          TYPE_REIMS_VGPU_PCI, rc);
        }
        qemu_bh_schedule(s->action_bh);
    }
    return NULL;
}

/*
 * Oversample the Rust-owned VBL limiter (DISPLAY_VBL_MIN_INTERVAL_MS = 8 ms for
 * the 120 Hz advertised mode). Polling at 4 ms guarantees the 8 ms limiter is
 * the gate even when the main loop stalls under input/QMP/display work. This
 * thread only supplies poll opportunities and schedules the existing main-loop
 * BH; Rust owns pacing and protocol state, while the BH remains the sole
 * HostAction applier.
 */
#define REIMS_VGPU_PCI_HEARTBEAT_MS 4

static void *reims_vgpu_pci_heartbeat_thread(void *opaque)
{
    ReimsVGPUPCIState *s = opaque;

    qemu_mutex_lock(&s->heartbeat_mutex);
    while (!s->heartbeat_stopping) {
        qemu_cond_timedwait(&s->heartbeat_cond, &s->heartbeat_mutex,
                            REIMS_VGPU_PCI_HEARTBEAT_MS);
        if (s->heartbeat_stopping) {
            break;
        }
        qemu_mutex_unlock(&s->heartbeat_mutex);

        if (s->rust_handle != 0 &&
            reims_vgpu_qemu_device_poll(s->rust_handle) == REIMS_VGPU_QEMU_OK &&
            s->action_bh) {
            qemu_bh_schedule(s->action_bh);
        }

        qemu_mutex_lock(&s->heartbeat_mutex);
    }
    qemu_mutex_unlock(&s->heartbeat_mutex);
    return NULL;
}

static bool reims_vgpu_pci_fb_update(void *opaque)
{
    ReimsVGPUPCIState *s = opaque;
    uint32_t kind, mid = 0, w = 0, h = 0, gen = 0;

    if (!s->con) {
        return true;
    }

    if (s->rust_handle != 0 &&
        reims_vgpu_qemu_device_poll(s->rust_handle) == REIMS_VGPU_QEMU_OK) {
        reims_vgpu_pci_deliver_actions(s);
    }

    /* Host-console ownership is Rust's call; this only paints what it names. */
    kind = reims_vgpu_shim_console_feed(s->rust_handle, &mid, &w, &h, &gen);

    if (kind == REIMS_VGPU_CONSOLE_FEED_EARLY) {
        /* Re-pull the latched front (archive fb_update early path).
         *
         * Return either way, for the same reason the firmware arm below states:
         * a failed paint must NOT fall through to the re-push, whose pending
         * frame is a product one. _EARLY means Rust says the early console owns
         * the screen, so pushing a product frame here is the same pre-boundary
         * steal, and this arm used to do it while the arm64 shim returned.
         *
         * No console update here: unlike the arm64 shim's, this
         * `paint_scanout` already calls `qemu_console_update_full` and clears
         * `new_frame_ready` itself on the path that returns true. */
        (void)reims_vgpu_pci_paint_scanout(s, mid, w, h, gen);
        return true;
    } else if (kind == REIMS_VGPU_CONSOLE_FEED_FIRMWARE) {
        if (reims_vgpu_pci_copy_early_console(s)) {
            qemu_console_update_full(s->con);
            s->new_frame_ready = false;
        }
        /* Return either way. A failed firmware copy must NOT fall through to
         * the re-push below: the pending frame there is a product one, and
         * pushing it while Rust says the firmware console owns the screen is
         * the pre-boundary steal this feed exists to prevent. */
        return true;
    }

    /* Nothing painted this tick — re-push the last frame if one is pending.
     * _PRODUCT reaches here every tick: apply_scanout does that painting. */
    if (s->new_frame_ready && s->surface) {
        qemu_console_update_full(s->con);
        s->new_frame_ready = false;
    }
    return true;
}

static const GraphicHwOps reims_vgpu_pci_fb_ops = {
    .gfx_update = reims_vgpu_pci_fb_update,
};

/* ---------- MMIO (forward only) ---------- */

static uint64_t reims_vgpu_pci_gfx_read(void *opaque, hwaddr offset, unsigned size)
{
    ReimsVGPUPCIState *s = opaque;
    uint64_t val = 0;

    if (s->rust_handle == 0) {
        return 0;
    }
    if (reims_vgpu_qemu_gfx_read(s->rust_handle, offset, size, &val) != REIMS_VGPU_QEMU_OK) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: gfx read failed offset=0x%" HWADDR_PRIx " size=%u\n",
                      TYPE_REIMS_VGPU_PCI, offset, size);
        return 0;
    }
    trace_reims_vgpu_pci_gfx_read(offset, val);
    return val;
}

static void reims_vgpu_pci_gfx_write(void *opaque, hwaddr offset, uint64_t data,
                              unsigned size)
{
    ReimsVGPUPCIState *s = opaque;

    if (s->rust_handle == 0) {
        return;
    }
    trace_reims_vgpu_pci_gfx_write(offset, data);
    /*
     * Before the register write, not after: this is the guest handing the
     * device work, so every guest store ordered before the handoff must be
     * observed before anything that work does can reuse a host-side copy of
     * those pages. Harvesting here is also the only place it can happen — the
     * accelerator's dirty-log sync needs the BQL, which a vCPU MMIO write
     * holds and the drain thread must never take.
     *
     * Cheap when nothing is tracked or when nothing has read a generation
     * since the last harvest, so a burst of register writes costs one sync.
     */
    reims_vgpu_dirty_harvest(s->dirty);
    if (reims_vgpu_qemu_gfx_write(s->rust_handle, offset, data, size) != REIMS_VGPU_QEMU_OK) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: gfx write failed offset=0x%" HWADDR_PRIx
                      " data=0x%" PRIx64 " size=%u\n",
                      TYPE_REIMS_VGPU_PCI, offset, data, size);
        return;
    }
    /*
     * Deliver residual HostActions on the MMIO path, but do not run heavy
     * GPU work under BQL here — Rust schedules BH for pure-host work after
     * snapshotting guest inputs.
     */
    reims_vgpu_pci_deliver_actions(s);
}

static const MemoryRegionOps reims_vgpu_pci_gfx_ops = {
    .read = reims_vgpu_pci_gfx_read,
    .write = reims_vgpu_pci_gfx_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 8,
    },
    .impl = {
        .min_access_size = 4,
        .max_access_size = 8,
    },
};

/* ---------- Lifecycle ---------- */

/*
 * Stop and destroy the Rust/Vulkan backend before QEMU enters process/driver
 * teardown.  PCI exit is not guaranteed to run on the main-loop shutdown
 * path: a guest panic previously left reims-vgpu-pci-drain inside Vulkan execution
 * while libc exit unloaded NVIDIA EGL, crashing QEMU in libnvidia-eglcore.
 */
static void reims_vgpu_pci_stop_backend(ReimsVGPUPCIState *s)
{
    if (s->heartbeat_started) {
        qemu_mutex_lock(&s->heartbeat_mutex);
        s->heartbeat_stopping = true;
        qemu_cond_signal(&s->heartbeat_cond);
        qemu_mutex_unlock(&s->heartbeat_mutex);
        qemu_thread_join(&s->heartbeat_thread);
        s->heartbeat_started = false;
    }
    if (s->drain_started) {
        qemu_mutex_lock(&s->drain_mutex);
        s->drain_stopping = true;
        qemu_cond_signal(&s->drain_cond);
        qemu_mutex_unlock(&s->drain_mutex);
        qemu_thread_join(&s->drain_thread);
        s->drain_started = false;
    }
    if (s->action_bh) {
        qemu_bh_delete(s->action_bh);
        s->action_bh = NULL;
    }
    if (s->rust_handle != 0) {
        /*
         * Close + join the host window first (no-op if none): its Vulkan
         * objects tear down on the window thread before we proceed, so the
         * process/driver-unload teardown never races live Vulkan work (the
         * libnvidia-eglcore crash class the drain teardown already guards).
         */
        reims_vgpu_qemu_window_stop(s->rust_handle);
        reims_vgpu_qemu_device_destroy(s->rust_handle);
        s->rust_handle = 0;
    }
}

static void reims_vgpu_pci_shutdown_notifier(Notifier *n, void *data)
{
    ReimsVGPUPCIState *s = container_of(n, ReimsVGPUPCIState,
                                         shutdown_notifier);

    (void)data;
    reims_vgpu_pci_stop_backend(s);
}

static void reims_vgpu_pci_realize(PCIDevice *pdev, Error **errp)
{
    ReimsVGPUPCIState *s = REIMS_VGPU_PCI(pdev);
    ReimsVgpuQemuCreateInfo info;
    ReimsVgpuQemuDevice out = {
        .abi_version = 0,
        .struct_size = 0,
        .handle = 0,
    };
    char backend[32];
    int rc;

    if (reims_vgpu_qemu_abi_version() != REIMS_VGPU_QEMU_ABI_VERSION) {
        error_setg(errp,
                   "%s: ABI version mismatch (header %u, staticlib %u)",
                   TYPE_REIMS_VGPU_PCI, REIMS_VGPU_QEMU_ABI_VERSION,
                   reims_vgpu_qemu_abi_version());
        return;
    }

    memory_region_init_io(&s->iomem_gfx, OBJECT(s), &reims_vgpu_pci_gfx_ops, s,
                          TYPE_REIMS_VGPU_PCI ".gfx",
                          REIMS_VGPU_GFX_MMIO_SIZE);
    /* 32-bit non-prefetch BAR (16 KiB control window). Live 2026-07-13: 64-bit
     * BAR behind pcie-root-port caused Apple efiboot STOP 0x15; keep 32-bit. */
    pci_register_bar(pdev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->iomem_gfx);

    /*
     * BAR1: host RAM linear framebuffer for UEFI Graphics Output Protocol.
     * OpenCore/OVMF have no built-in driver for 0x106B:0xEEEE; our PCI option
     * ROM (or OC Drivers entry) installs GOP with FrameBufferBase = BAR1.
     * Prefetchable 32-bit MEM so OVMF assigns under 4G (same STOP 0x15 caution).
     */
    memory_region_init_ram(&s->fb_vram, OBJECT(s), TYPE_REIMS_VGPU_PCI ".fb",
                           REIMS_VGPU_PCI_FB_SIZE, &error_fatal);
    /*
     * Non-prefetchable 32-bit MEM (same as BAR0 caution). Prefetchable BAR1
     * was a live suspect for macOS "console relocated" off the display BAR
     * into system RAM (0xf1000000) while VMware SVGA keeps console on its
     * VRAM BAR at the same GPA. Host can only follow system-RAM consoles via
     * EFI FB start (0x1210); pure XNU relocate does not rewrite that reg.
     */
    pci_register_bar(pdev, 1, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->fb_vram);

    /*
     * Behind pcie-root-port the endpoint must be a real PCIe function with an
     * Express capability. Conventional-only devices leave the secondary bus
     * empty under macOS (guest ioreg: bridge 0:5 only, no 0x106B:0xEEEE).
     * Mirror bochs-display / virtio-pci.
     */
    if (pci_bus_is_express(pci_get_bus(pdev))) {
        rc = pcie_endpoint_cap_init(pdev, 0x80);
        if (rc < 0) {
            error_setg(errp, "%s: pcie_endpoint_cap_init failed",
                       TYPE_REIMS_VGPU_PCI);
            return;
        }
    } else {
        pdev->cap_present &= ~QEMU_PCI_CAP_EXPRESS;
    }

    /* Guest enables MSI count 1; always raise vector 0. */
    rc = msi_init(pdev, 0x0, 1, true /* 64-bit */, false /* per-vector mask */,
                  errp);
    if (rc < 0) {
        return;
    }

    s->host_ops = (ReimsVgpuHostOps){
        .abi_version = REIMS_VGPU_QEMU_ABI_VERSION,
        .struct_size = sizeof(ReimsVgpuHostOps),
        .ctx = s,
        .read_gpa = reims_vgpu_shim_read_gpa,
        .write_gpa = reims_vgpu_shim_write_gpa,
        .mono_ns = reims_vgpu_shim_mono_ns,
        .schedule_bh = reims_vgpu_pci_schedule_bh,
        .read_kva = reims_vgpu_shim_read_kva,
        .read_xreg = reims_vgpu_pci_read_xreg,
        .map_pages = reims_vgpu_pci_map_pages,
        .unmap_pages = reims_vgpu_pci_unmap_pages,
        /*
         * Where guest RAM lives. Shared with the sysbus shim and identical on
         * both: it reads the system address space, which the attach type does
         * not change. Unlike map_pages it never builds a view, so there is no
         * bus-specific lifetime for a shim to own.
         */
        .guest_ram_regions = reims_vgpu_shim_guest_ram_regions,
        .is_ram_gpa = reims_vgpu_shim_is_ram_gpa,
        .notify_actions = reims_vgpu_pci_notify_actions,
        /*
         * Every result remains valid until the caller explicitly retires it:
         * direct RAMBlock pointers need no release, while packed aliases leave
         * through unmap_pages after their Vulkan import has retired.
         */
        .map_pages_stable = 1,
        .track_guest_writes = reims_vgpu_pci_track_guest_writes,
        .untrack_guest_writes = reims_vgpu_pci_untrack_guest_writes,
        .guest_write_gen = reims_vgpu_pci_guest_write_gen,
        .guest_written_pages = reims_vgpu_pci_guest_written_pages,
        .page_alias_census = reims_vgpu_pci_page_alias_census,
    };
    s->dirty = reims_vgpu_dirty_new();

    qemu_mutex_init(&s->drain_mutex);
    qemu_cond_init(&s->drain_cond);
    qemu_mutex_init(&s->heartbeat_mutex);
    qemu_cond_init(&s->heartbeat_cond);
    s->action_bh = aio_bh_new(qemu_get_aio_context(), reims_vgpu_pci_bh, s);

    info = (ReimsVgpuQemuCreateInfo){
        .abi_version = REIMS_VGPU_QEMU_ABI_VERSION,
        .struct_size = sizeof(ReimsVgpuQemuCreateInfo),
        .host_ops = &s->host_ops,
        /* x86 Tahoe guest: 4 KiB pages. */
        .guest_page_shift = REIMS_VGPU_GUEST_PAGE_SHIFT_X86_64,
    };

    rc = reims_vgpu_qemu_device_create(&info, &out);
    if (rc != REIMS_VGPU_QEMU_OK || out.handle == 0) {
        error_setg(errp, "%s: reims_vgpu_qemu_device_create failed (rc=%d)",
                   TYPE_REIMS_VGPU_PCI, rc);
        msi_uninit(pdev);
        qemu_bh_delete(s->action_bh);
        s->action_bh = NULL;
        qemu_cond_destroy(&s->heartbeat_cond);
        qemu_mutex_destroy(&s->heartbeat_mutex);
        qemu_cond_destroy(&s->drain_cond);
        qemu_mutex_destroy(&s->drain_mutex);
        return;
    }
    s->rust_handle = out.handle;
    qemu_thread_create(&s->drain_thread, "reims-vgpu-pci-drain",
                       reims_vgpu_pci_drain_thread, s, QEMU_THREAD_JOINABLE);
    s->drain_started = true;
    qemu_thread_create(&s->heartbeat_thread, "reims-vgpu-pci-heartbeat",
                       reims_vgpu_pci_heartbeat_thread, s, QEMU_THREAD_JOINABLE);
    s->heartbeat_started = true;
    s->shutdown_notifier.notify = reims_vgpu_pci_shutdown_notifier;
    qemu_register_shutdown_notifier(&s->shutdown_notifier);
    s->shutdown_notifier_registered = true;

    s->con = qemu_graphic_console_create(DEVICE(pdev), 0, &reims_vgpu_pci_fb_ops, s);
    reims_vgpu_pci_set_mode(s, REIMS_VGPU_EFI_BOOT_WIDTH, REIMS_VGPU_EFI_BOOT_HEIGHT);
    if (s->surface) {
        memset(surface_data(s->surface), 0,
               (size_t)surface_stride(s->surface) * REIMS_VGPU_EFI_BOOT_HEIGHT);
        qemu_console_update_full(s->con);
    }
    if (s->con) {
        qemu_console_set_cursor(s->con, cursor_builtin_hidden());
        qemu_console_set_mouse(s->con, 0, 0, false);
    }

    /*
     * Host-owned presentation window (kb host-window): on unless REIMS_VGPU_WINDOW
     * disables it (reims_vgpu_pci_window_requested; vm/boot-x86.sh defaults it on for
     * this device). Rust spawns a winit + VkSurfaceKHR window on its own thread,
     * the drain publishes finished present frames to it, and window input is
     * injected
     * through the neutral Input* prompt-action rail (qemu_input_*, which works
     * under -display none). All window/present logic lives in the staticlib;
     * the shim only flips it on. A staticlib built without the host-window
     * feature returns ERR_STATE here and QEMU's own display stays in charge.
     */
    if (reims_vgpu_pci_window_requested()) {
        int wrc = reims_vgpu_qemu_window_start(s->rust_handle, REIMS_VGPU_EFI_BOOT_WIDTH,
                                        REIMS_VGPU_EFI_BOOT_HEIGHT);
        if (wrc != REIMS_VGPU_QEMU_OK) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "%s: host window unavailable (rc=%d); "
                          "using QEMU display\n",
                          TYPE_REIMS_VGPU_PCI, wrc);
        } else if (memory_region_is_ram(&s->fb_vram)) {
            /*
             * Hand the window the BAR1 GOP framebuffer (host RAM) so it shows
             * UEFI/OpenCore/boot.efi output before the product present path
             * latches. The pointer is a stable RAMBlock host VA valid for the
             * device lifetime; the staticlib reads it on the poll path (gated
             * by protocol console-ownership state, never content).
             */
            uint8_t *fb = memory_region_get_ram_ptr(&s->fb_vram);

            if (fb) {
                reims_vgpu_qemu_window_set_early_fb(
                    s->rust_handle, fb,
                    REIMS_VGPU_EFI_BOOT_WIDTH * REIMS_VGPU_PCI_EFI_BPP,
                    REIMS_VGPU_EFI_BOOT_WIDTH, REIMS_VGPU_EFI_BOOT_HEIGHT);
            }
        }
    }

    backend[0] = '\0';
    reims_vgpu_qemu_backend_name(backend, sizeof(backend));
    trace_reims_vgpu_pci_realize(s->rust_handle, backend);
}

static void reims_vgpu_pci_exit(PCIDevice *pdev)
{
    ReimsVGPUPCIState *s = REIMS_VGPU_PCI(pdev);

    if (s->shutdown_notifier_registered) {
        notifier_remove(&s->shutdown_notifier);
        s->shutdown_notifier_registered = false;
    }
    reims_vgpu_pci_stop_backend(s);
    if (s->page_views) {
        size_t i;

        for (i = 0; i < s->page_views->len; i++) {
            ReimsVGPUPCIPageView *view =
                &g_array_index(s->page_views, ReimsVGPUPCIPageView, i);
#ifndef _WIN32
            munmap(view->ptr, view->len);
            s->page_views_destroyed++;
#endif
        }
        g_array_free(s->page_views, true);
        s->page_views = NULL;
    }
    /* After the drain thread is joined: reims_vgpu_dirty_free turns region
     * logging back off, and no tracked set may outlive the Rust device that
     * holds its token. */
    reims_vgpu_dirty_free(s->dirty);
    s->dirty = NULL;
    qemu_cond_destroy(&s->heartbeat_cond);
    qemu_mutex_destroy(&s->heartbeat_mutex);
    qemu_cond_destroy(&s->drain_cond);
    qemu_mutex_destroy(&s->drain_mutex);
    msi_uninit(pdev);
    if (s->con) {
        qemu_graphic_console_close(s->con);
        s->con = NULL;
    }
    s->surface = NULL;
}

static void reims_vgpu_pci_reset(DeviceState *dev)
{
    ReimsVGPUPCIState *s = REIMS_VGPU_PCI(dev);

    if (s->rust_handle != 0) {
        reims_vgpu_qemu_device_reset(s->rust_handle);
    }
    qemu_mutex_lock(&s->drain_mutex);
    s->drain_pending = false;
    qemu_mutex_unlock(&s->drain_mutex);
    s->new_frame_ready = false;
    reims_vgpu_pci_set_mode(s, REIMS_VGPU_EFI_BOOT_WIDTH, REIMS_VGPU_EFI_BOOT_HEIGHT);
    if (s->surface && s->con) {
        memset(surface_data(s->surface), 0,
               (size_t)surface_stride(s->surface) * REIMS_VGPU_EFI_BOOT_HEIGHT);
        qemu_console_set_cursor(s->con, cursor_builtin_hidden());
        qemu_console_set_mouse(s->con, 0, 0, false);
        s->new_frame_ready = true;
        qemu_console_update_full(s->con);
        s->new_frame_ready = false;
    }
}

static void reims_vgpu_pci_instance_init(Object *obj)
{
    ReimsVGPUPCIState *s = REIMS_VGPU_PCI(obj);
    PCIDevice *pdev = PCI_DEVICE(obj);

    /* Required before realize so config space is PCIe-sized (4 KiB). */
    pdev->cap_present |= QEMU_PCI_CAP_EXPRESS;
    s->page_views = g_array_new(false, false, sizeof(ReimsVGPUPCIPageView));
}

static void reims_vgpu_pci_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    (void)data;
    k->realize = reims_vgpu_pci_realize;
    k->exit = reims_vgpu_pci_exit;
    k->vendor_id = REIMS_VGPU_PCI_VENDOR;
    k->device_id = REIMS_VGPU_PCI_DEVICE;
    k->revision = 0x00;
    k->class_id = PCI_CLASS_DISPLAY_VGA;
    k->subsystem_vendor_id = REIMS_VGPU_PCI_VENDOR;
    k->subsystem_id = REIMS_VGPU_PCI_DEVICE;

    dc->desc = "Reims vGPU (PCI thin shim -> reims-vgpu)";
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);
    dc->hotpluggable = false;
    dc->user_creatable = true;
    device_class_set_legacy_reset(dc, reims_vgpu_pci_reset);
}

static const TypeInfo reims_vgpu_pci_info = {
    .name = TYPE_REIMS_VGPU_PCI,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(ReimsVGPUPCIState),
    .instance_init = reims_vgpu_pci_instance_init,
    .class_init = reims_vgpu_pci_class_init,
    .interfaces = (const InterfaceInfo[]) {
        { INTERFACE_PCIE_DEVICE },
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void reims_vgpu_pci_register_types(void)
{
    type_register_static(&reims_vgpu_pci_info);
}

type_init(reims_vgpu_pci_register_types)
