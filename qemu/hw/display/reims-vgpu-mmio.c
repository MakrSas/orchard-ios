/*
 * QEMU thin shim for the product paravirtualized GPU MMIO device.
 *
 * Sibling of apple-gfx-mmio: same sysbus layout on the vmapple machine
 * (mmio 0 / irq 0 = gfx window, mmio 1 / irq 1 = IOSurface mapper). The guest
 * binds Apple's own AppleParavirtGPU.kext; this host device is selected with
 * -M vmapple,gfx-device=reims-vgpu-mmio.
 *
 * Deliberately no more than apple-gfx's wrapper around
 * ParavirtualizedGraphics.framework:
 *   - SysBus registration + MemoryRegionOps
 *   - HostOps callbacks (GPA/KVA R/W, xreg, clock, schedule BH)
 *   - oneshot BH: drain Rust + apply HostActions (IRQ / scanout / cursor)
 *   - GraphicHwOps console surface (mode + update_full); pixels from Rust
 *
 * Protocol, FIFO, decode, mapper capture, and GPU work all live in Rust.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/memalign.h"
#include "qemu/module.h"
#include "qemu/timer.h"
#include "qemu/thread.h"
#include "qemu/main-loop.h"
#include "qemu/aio.h"
#include "qapi/error.h"
#include "qemu-main.h"
#include "hw/core/cpu.h"
#include "hw/core/sysbus.h"
#include "hw/core/irq.h"
#include "qom/object.h"
#include "qapi/visitor.h"
#include "qapi/error.h"
#include "system/address-spaces.h"
#include "system/hw_accel.h"
#include "system/memory.h"
/*
 * RAMBlock queries (`qemu_ram_block_from_host`, `qemu_ram_is_shared`) live here
 * now. Without the declaration C defaults the first to returning `int`, which
 * truncates the RAMBlock pointer to 32 bits and segfaults the moment it is
 * dereferenced -- a warning, not an error, so the build stays green and the
 * crash lands 80 seconds into a guest boot.
 */
#include "system/ramblock.h"
#include "system/runstate.h"
#include "ui/console.h"
#include "ui/surface.h"
#include "trace.h"
#include "reims_vgpu_qemu_abi.h"
#include "reims-vgpu-dirty.h"
#include "reims-vgpu-shim.h"
#ifdef CONFIG_ORCHARD_EMBED
#include "ui/orchard-embed.h"
#endif

/*
 * `qemu_graphic_console_create` and the `qemu_console_*` setters are the
 * current upstream spelling of what used to be `graphic_console_init` and the
 * `dpy_*` calls. This file uses the new names directly.
 */

/*
 * Guest X-regs and mach_vm page aliasing are Darwin product paths (arm guest
 * under HVF). TARGET_* macros are poisoned in common softmmu objects, so these
 * paths are gated on CONFIG_DARWIN only. On Linux/x86 the callbacks fail closed;
 * protocol/decode still runs through the Rust staticlib (Metal host stubs).
 */
#if defined(CONFIG_DARWIN)
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
/*
 * The iOS SDK refuses <mach/mach_vm.h> outright ("mach_vm.h unsupported"),
 * but a task may still remap its own pages through the older vm_* calls in
 * <mach/vm_map.h> — the same calls an iOS JIT uses to mirror its code buffer.
 * On a 64-bit task vm_address_t and mach_vm_address_t describe the same
 * addresses; they are only distinct C types, hence the copies through locals
 * rather than a rename.
 */
#include <mach/vm_map.h>

static inline kern_return_t reims_vm_allocate(vm_map_t task, mach_vm_address_t *addr,
                                              mach_vm_size_t size, int flags)
{
    vm_address_t a = (vm_address_t)*addr;
    kern_return_t kr = vm_allocate(task, &a, (vm_size_t)size, flags);

    *addr = a;
    return kr;
}

static inline kern_return_t reims_vm_remap(vm_map_t task, mach_vm_address_t *dst,
                                           mach_vm_size_t size, mach_vm_offset_t mask,
                                           int flags, vm_map_t src_task,
                                           mach_vm_address_t src, boolean_t copy,
                                           vm_prot_t *cur, vm_prot_t *max,
                                           vm_inherit_t inherit)
{
    vm_address_t d = (vm_address_t)*dst;
    kern_return_t kr = vm_remap(task, &d, (vm_size_t)size, (vm_address_t)mask, flags,
                                src_task, (vm_address_t)src, copy, cur, max, inherit);

    *dst = d;
    return kr;
}

static inline kern_return_t reims_vm_deallocate(vm_map_t task, mach_vm_address_t addr,
                                                mach_vm_size_t size)
{
    return vm_deallocate(task, (vm_address_t)addr, (vm_size_t)size);
}

#define mach_vm_allocate   reims_vm_allocate
#define mach_vm_remap      reims_vm_remap
#define mach_vm_deallocate reims_vm_deallocate
#else
#include <mach/mach_vm.h>
#endif
#endif
#if defined(TARGET_AARCH64) || defined(TARGET_ARM)
#include "target/arm/cpu.h"
#endif
#include <sys/mman.h>

#define TYPE_REIMS_VGPU_MMIO "reims-vgpu-mmio"
OBJECT_DECLARE_SIMPLE_TYPE(ReimsVGPUMMIOState, REIMS_VGPU_MMIO)

/*
 * Window sizes match the live Reims VGPU contract / apple-gfx-mmio:
 * gfx = 16 KiB, iosfc = 64 KiB. The gfx size is the shared
 * REIMS_VGPU_GFX_MMIO_SIZE — Rust bounds its register store against the same
 * number and a private copy here is a window the guest can address past it.
 * The iosfc size stays local; Rust keeps no per-offset state for that rail, so
 * mirroring it would create a source of truth nothing checks.
 */
#define REIMS_VGPU_MMIO_IOSFC_MMIO_SIZE 0x10000

/* Rust device/window action poll cadence (250 Hz, non-blocking). */
#define REIMS_VGPU_MMIO_WINDOW_POLL_MS 4

/*
 * One packed mach_vm_remap view handed out by map_pages, and the length it was
 * allocated at. The length is recorded because unmap_pages cannot trust the
 * caller's: a run whose first page is entered at an offset asks to release
 * fewer bytes than the view spans, and mach_vm_deallocate takes the allocation.
 */
typedef struct ReimsVGPUMMIOPageView {
    void *ptr;
    size_t len;
} ReimsVGPUMMIOPageView;

struct ReimsVGPUMMIOState {
    SysBusDevice parent_obj;

    MemoryRegion iomem_gfx;
    MemoryRegion iomem_iosfc;
    qemu_irq irq_gfx;
    qemu_irq irq_iosfc;

    /* Display (apple-gfx console role only). */
    QemuConsole *con;
    DisplaySurface *surface;
    /*
     * Guest frame-ready (archive apple-pv-gpu present-boundary policy +
     * apple-gfx new_frame_ready). Set when CmdDisplaySwap / early front paint
     * has written a finished frame into `surface`. gfx_update only pushes the
     * console when this is set — never host fixed-rate re-pull of live guest
     * pages (that dual-mid A/B thrash / dock-band mid-composite).
     */
    bool new_frame_ready;
    /* Device poll stays on QEMU's background emulation loop. */
    QEMUTimer *poll_timer;
    QEMUBH *action_bh;
    /*
     * The drain worker. Rust asks for a drain through schedule_bh; it used to
     * run as a main-loop BH, which is to say under the BQL, and a drain that
     * encodes a frame holds it for as long as that takes — measured on an
     * iPhone under TCG, chains of 84-104 ms. Every vCPU that touched a device
     * in that time (the GIC, the USB controller, a timer) waited for it, while
     * the guest's clock ran on: frames lost, and a key's release read so late
     * that the guest repeated the key. The PCI shim has always drained on a
     * thread of its own; this does the same. Nothing the drain calls needs the
     * BQL: guest memory goes through address_space_rw, the dirty tracker has
     * its own lock, and HostActions still reach QEMU through action_bh.
     */
    QemuThread drain_thread;
    QemuMutex drain_mutex;
    QemuCond drain_cond;
    bool drain_pending;
    bool drain_stopping;
    bool drain_started;

    /* Early boot framebuffer registered before/during realize. */
    const uint8_t *early_fb_ptr;
    size_t early_fb_stride;
    uint32_t early_fb_width;
    uint32_t early_fb_height;

    /* Filled at realize; address passed into Rust create. */
    ReimsVgpuHostOps host_ops;
    /* Hypervisor dirty-bitmap adapter: the only witness for a write to a
     * surface's guest pages that no device operation made. */
    ReimsVgpuDirty *dirty;
    /* Opaque handle from reims_vgpu_qemu_device_create; 0 when unrealized. */
    uint64_t rust_handle;
    /*
     * Live packed mach_vm_remap views, so unmap_pages can tell one of ours
     * from a direct RAMBlock HVA — the two are indistinguishable as bare
     * pointers, and mach_vm_deallocate on the latter would unmap guest RAM
     * out from under the VM. Entries are removed as they are released; a
     * non-empty array at teardown is a leak, and reims_vgpu_mmio_free_page_views
     * says so.
     */
    GArray *page_views;
    /*
     * Guards page_views. map_pages and unmap_pages run on the drain worker as
     * well as on vCPU threads, and the BQL that once serialized them is no
     * longer held by the worker.
     */
    QemuMutex page_views_lock;
};

static ReimsVGPUMMIOState *reims_vgpu_mmio_instance;

/*
 * Not in the library build: there is no QEMU main() to hand over there, and the
 * embedding app already owns its initial thread (see include/ui/orchard-embed.h).
 */
#if defined(CONFIG_DARWIN) && !defined(CONFIG_ORCHARD_EMBED)
/*
 * winit/AppKit owns the initial process thread. QEMU's Darwin main wrapper
 * already moves its emulation loop to a background thread when qemu_main is
 * non-NULL; device realize runs before that handoff and creates the event loop
 * on the same initial thread.
 */
static ReimsVGPUMMIOState *reims_vgpu_mmio_window_owner;

static int reims_vgpu_mmio_window_main_loop(void)
{
    ReimsVGPUMMIOState *s = reims_vgpu_mmio_window_owner;
    int rc;

    if (!s || s->rust_handle == 0) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: host window main loop has no live owner\n",
                      TYPE_REIMS_VGPU_MMIO);
        dispatch_main();
        g_assert_not_reached();
    }

    rc = reims_vgpu_qemu_window_run_main(s->rust_handle);
    if (rc != REIMS_VGPU_QEMU_OK) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: host window main loop failed rc=%d\n",
                      TYPE_REIMS_VGPU_MMIO, rc);
    }

    /*
     * The background QEMU loop owns shutdown and calls exit(). Keep the initial
     * thread available to libdispatch after the winit loop has closed.
     */
    dispatch_main();
    g_assert_not_reached();
}
#endif

/* ---------- HostOps (only host services; apple-gfx equivalents) ---------- */

/*
 * Guest X-register read for the iosfc mapper directed handoff
 * (x19 mapper device, x21 request type, x22 MappingInternal*). Must be
 * invoked on the MMIO path of the publishing vCPU — Rust calls this from
 * iosfc producer write (sync path). No first_cpu fallback: same deadlock
 * class as read_kva if used from the BH.
 */
static int reims_vgpu_mmio_read_xreg(void *ctx, uint32_t index, uint64_t *out)
{
#if defined(TARGET_AARCH64) || defined(TARGET_ARM)
    CPUState *cs = current_cpu;
    ARMCPU *cpu;

    if (!out || index >= 32) {
        return -1;
    }
    if (!cs) {
        return -1;
    }
    cpu_synchronize_state(cs);
    cpu = ARM_CPU(cs);
    *out = cpu->env.xregs[index];
    return 0;
#else
    (void)ctx;
    (void)index;
    (void)out;
    return -1;
#endif
}

/*
 * One host-contiguous view over a fragmented run of guest pages, which the
 * caller owns and releases through unmap_pages. Takes ownership of `hvas`
 * either way.
 */
static int reims_vgpu_pack_fragmented_view(ReimsVGPUMMIOState *s,
                                           uint8_t **hvas, size_t count,
                                           void **out_ptr,
                                           ReimsVgpuMapPagesFailure *failure)
{
    size_t i;

#if defined(CONFIG_DARWIN)
    {
        mach_vm_address_t view = 0;
        mach_vm_size_t view_len = (mach_vm_size_t)count * REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E;
        ReimsVGPUMMIOPageView view_entry;
        kern_return_t kr;

        kr = mach_vm_allocate(mach_task_self(), &view, view_len,
                              VM_FLAGS_ANYWHERE);
        if (kr != KERN_SUCCESS) {
            failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_RESERVATION;
            g_free(hvas);
            return -1;
        }
        for (i = 0; i < count; i++) {
            mach_vm_address_t dst = view + i * REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E;
            vm_prot_t cur_prot, max_prot;

            kr = mach_vm_remap(mach_task_self(), &dst, REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E, 0,
                               VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
                               mach_task_self(),
                               (mach_vm_address_t)(uintptr_t)hvas[i], FALSE,
                               &cur_prot, &max_prot, VM_INHERIT_NONE);
            if (kr != KERN_SUCCESS) {
                failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_ALIAS;
                failure->page_index = i;
                mach_vm_deallocate(mach_task_self(), view, view_len);
                g_free(hvas);
                return -1;
            }
        }

        *out_ptr = (void *)(uintptr_t)view;
        view_entry.ptr = *out_ptr;
        view_entry.len = view_len;
        qemu_mutex_lock(&s->page_views_lock);
        g_array_append_val(s->page_views, view_entry);
        qemu_mutex_unlock(&s->page_views_lock);
        g_free(hvas);
        return 0;
    }
#elif !defined(_WIN32)
    {
        size_t total = count * REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E;
        uint8_t *view = mmap(NULL, total, PROT_NONE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ReimsVGPUMMIOPageView held;

        if (view == MAP_FAILED) {
            failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_RESERVATION;
            failure->host_errno = errno;
            g_free(hvas);
            return -1;
        }
        rcu_read_lock();
        for (i = 0; i < count; i++) {
            RAMBlock *rb;
            ram_addr_t rb_offset;
            ram_addr_t fd_offset;
            int fd;
            void *mapped;

            rb = qemu_ram_block_from_host(hvas[i], false, &rb_offset);
            if (!rb || !qemu_ram_is_shared(rb)) {
                failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_ALIAS;
                failure->page_index = i;
                goto alias_fail_linux;
            }
            fd = qemu_ram_get_fd(rb);
            if (fd < 0 || rb_offset > RAM_ADDR_MAX - qemu_ram_get_fd_offset(rb)) {
                failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_ALIAS;
                failure->page_index = i;
                goto alias_fail_linux;
            }
            fd_offset = qemu_ram_get_fd_offset(rb) + rb_offset;
            if ((fd_offset & (REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E - 1)) != 0) {
                failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_ALIAS;
                failure->page_index = i;
                goto alias_fail_linux;
            }
            mapped = mmap(view + i * REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E,
                          REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E,
                          PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_FIXED, fd, fd_offset);
            if (mapped == MAP_FAILED) {
                failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_ALIAS;
                failure->host_errno = errno;
                failure->page_index = i;
                goto alias_fail_linux;
            }
        }
        rcu_read_unlock();

        held.ptr = view;
        held.len = total;
        qemu_mutex_lock(&s->page_views_lock);
        g_array_append_val(s->page_views, held);
        qemu_mutex_unlock(&s->page_views_lock);
        g_free(hvas);
        *out_ptr = view;
        return 0;

alias_fail_linux:
        rcu_read_unlock();
        munmap(view, total);
        g_free(hvas);
        return -1;
    }
#else
    if (failure) {
        failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_RESERVATION;
    }
    g_free(hvas);
    return -1;
#endif
}

/*
 * The whole run in one host mapping, when the guest pages happen to be
 * sequential and land in one RAM region. NULL when they do not, which is the
 * caller's signal to take the per-page path; it is not a failure, so no failure
 * stage is recorded.
 *
 * Must be called under rcu_read_lock().
 */
static uint8_t *reims_vgpu_contiguous_run_hva(const uint64_t *gpas, size_t count)
{
    hwaddr total = (hwaddr)count * REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E;
    hwaddr xlat, plen = total;
    MemoryRegion *mr;
    uint8_t *hva;
    size_t i;

    /* count * PAGE and gpas[0] + (count - 1) * PAGE must not wrap. */
    if (count > UINT64_MAX / REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E ||
        gpas[0] > UINT64_MAX - (count - 1) * REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E) {
        return NULL;
    }
    for (i = 1; i < count; i++) {
        if (gpas[i] != gpas[0] + i * REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E) {
            return NULL;
        }
    }

    mr = address_space_translate(&address_space_memory, gpas[0], &xlat, &plen,
                                 true, MEMTXATTRS_UNSPECIFIED);
    if (!mr || !memory_region_is_ram(mr) || plen < total) {
        return NULL;
    }
    hva = (uint8_t *)memory_region_get_ram_ptr(mr) + xlat;
    if ((uintptr_t)hva & (REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E - 1)) {
        return NULL;
    }
    return hva;
}

/*
 * Resolve each guest page to its host address, in order. False on the first
 * page that does not resolve to a whole page-aligned RAM page, with `failure`
 * naming which one.
 *
 * Must be called under rcu_read_lock().
 */
static bool reims_vgpu_resolve_guest_hvas(const uint64_t *gpas, size_t count,
                                          uint8_t **hvas,
                                          ReimsVgpuMapPagesFailure *failure)
{
    size_t i;

    for (i = 0; i < count; i++) {
        hwaddr xlat, plen = REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E;
        MemoryRegion *mr;
        uint8_t *hva;

        mr = address_space_translate(&address_space_memory, gpas[i], &xlat,
                                     &plen, true, MEMTXATTRS_UNSPECIFIED);
        if (!mr || !memory_region_is_ram(mr) ||
            plen < REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E) {
            failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_INVALID_GUEST_PAGE;
            failure->page_index = i;
            return false;
        }
        hva = (uint8_t *)memory_region_get_ram_ptr(mr) + xlat;
        if ((uintptr_t)hva & (REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E - 1)) {
            failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_INVALID_GUEST_PAGE;
            failure->page_index = i;
            return false;
        }
        hvas[i] = hva;
    }
    return true;
}

/*
 * Contiguous host-VA view of guest 16 KiB pages — the ParavirtualizedGraphics
 * mapMemory model (mach_vm_remap of guest RAM into the framework's working
 * VA). The view aliases guest RAM: Metal render targets created on it write
 * guest memory directly, so there is exactly ONE copy of surface content.
 *
 * A host-contiguous page run returns its direct RAMBlock HVA, which is guest
 * RAM itself and outlives every view. A fragmented list gets one packed
 * mach_vm_remap view, which the caller owns and must release through
 * unmap_pages.
 *
 * On Darwin the view used to be retained for the whole device lifetime instead,
 * because Rust cached VK_EXT_external_memory_host imports over it and could
 * re-read the pages at any later point. Nothing imports guest pages on the
 * Metal rail now, so the retention bought nothing and every fragmented map
 * leaked a VA reservation until teardown. `map_pages_stable` is 0 there.
 *
 * Linux hosts get the same packed view through one MAP_SHARED mmap per page
 * over the RAMBlock's fd, so it needs shared file-backed guest RAM. Plain
 * anonymous RAM refuses at the alias stage (`qemu_map_pages_alias_failed`,
 * errno 0) and those maps take the copying rails instead. Backing RAM with
 * `-object memory-backend-memfd,share=on` is not the answer on an AMD host:
 * amdgpu's userptr rejects shared file mappings, so the whole-RAMBlock
 * VK_EXT_external_memory_host import then fails with
 * ERROR_INVALID_EXTERNAL_HANDLE, which costs more than the aliases buy.
 */
static int reims_vgpu_mmio_map_pages(void *ctx, const uint64_t *gpas,
                                  size_t count, void **out_ptr,
                                  ReimsVgpuMapPagesFailure *failure)
{
    ReimsVGPUMMIOState *s = ctx;
    uint8_t **hvas = NULL;
    uint8_t *hva;
    size_t i;

    if (failure) {
        *failure = (ReimsVgpuMapPagesFailure) {
            .stage = REIMS_VGPU_MAP_PAGES_FAILURE_NONE,
        };
    }
    if (!s || !gpas || count == 0 || !out_ptr || !failure ||
        count > SIZE_MAX / REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E) {
        if (failure) {
            failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_INVALID_GUEST_PAGE;
            failure->page_index = UINT64_MAX;
        }
        return -1;
    }

    rcu_read_lock();
    hva = reims_vgpu_contiguous_run_hva(gpas, count);
    if (hva) {
        rcu_read_unlock();
        *out_ptr = hva;
        return 0;
    }

    hvas = g_new(uint8_t *, count);
    if (!reims_vgpu_resolve_guest_hvas(gpas, count, hvas, failure)) {
        goto fail;
    }
    rcu_read_unlock();

    for (i = 1; i < count; i++) {
        if (hvas[i] != hvas[0] + i * REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E) {
            break;
        }
    }
    if (i == count) {
        *out_ptr = hvas[0];
        g_free(hvas);
        return 0;
    }

    return reims_vgpu_pack_fragmented_view(s, hvas, count, out_ptr, failure);

fail:
    rcu_read_unlock();
    g_free(hvas);
    return -1;
}

/*
 * Release a view map_pages handed out. A pointer that is not in `page_views` is
 * a direct RAMBlock HVA — guest RAM itself — and must be left alone; there is
 * nothing to free and deallocating it would unmap the guest's own memory.
 *
 * `len` is deliberately ignored in favour of the recorded allocation length.
 * The caller passes the span it asked for, and a run entered at a page offset
 * asks for fewer bytes than the view covers.
 */
static void reims_vgpu_mmio_unmap_pages(void *ctx, void *ptr, size_t len)
{
    ReimsVGPUMMIOState *s = ctx;
    size_t i;

    (void)len;
    if (!s || !ptr || !s->page_views) {
        return;
    }
    qemu_mutex_lock(&s->page_views_lock);
    for (i = 0; i < s->page_views->len; i++) {
        ReimsVGPUMMIOPageView *view =
            &g_array_index(s->page_views, ReimsVGPUMMIOPageView, i);
        if (view->ptr == ptr) {
#if defined(CONFIG_DARWIN)
            mach_vm_deallocate(mach_task_self(),
                               (mach_vm_address_t)(uintptr_t)ptr, view->len);
#elif !defined(_WIN32)
            munmap(view->ptr, view->len);
#endif
            g_array_remove_index_fast(s->page_views, i);
            break;
        }
    }
    qemu_mutex_unlock(&s->page_views_lock);
}

/*
 * Teardown backstop. Every view should already have been released by
 * unmap_pages, so anything still here is a caller that mapped and never freed —
 * reclaim it, but say so rather than reclaiming quietly, because the silent
 * version of this function is what hid the leak that made it necessary.
 */
static void reims_vgpu_mmio_free_page_views(ReimsVGPUMMIOState *s)
{
    size_t i;

    if (!s->page_views) {
        return;
    }
    if (s->page_views->len != 0) {
        qemu_log_mask(LOG_UNIMP,
                      "%s: %u guest page view(s) still mapped at teardown\n",
                      TYPE_REIMS_VGPU_MMIO, s->page_views->len);
    }
    for (i = 0; i < s->page_views->len; i++) {
        ReimsVGPUMMIOPageView *view =
            &g_array_index(s->page_views, ReimsVGPUMMIOPageView, i);
#if defined(CONFIG_DARWIN)
        mach_vm_deallocate(mach_task_self(),
                           (mach_vm_address_t)(uintptr_t)view->ptr,
                           view->len);
#elif !defined(_WIN32)
        munmap(view->ptr, view->len);
#endif
    }
    g_array_set_size(s->page_views, 0);
}


/*
 * Guest-write tracking. A surface's storage is plain guest RAM: the guest CPU
 * stores into it with no device operation, so nothing the Rust device counts
 * can witness such a store and every host-side copy of those pages is stale
 * from that instant. These three forward to the shared dirty-bitmap adapter.
 */
static uint64_t reims_vgpu_mmio_track_guest_writes(void *ctx, const uint64_t *gpas,
                                                   size_t count, size_t page_size)
{
    ReimsVGPUMMIOState *s = ctx;

    return reims_vgpu_dirty_track(s->dirty, gpas, count, page_size);
}

static void reims_vgpu_mmio_untrack_guest_writes(void *ctx, uint64_t token)
{
    ReimsVGPUMMIOState *s = ctx;

    reims_vgpu_dirty_untrack(s->dirty, token);
}

static uint64_t reims_vgpu_mmio_guest_write_gen(void *ctx, uint64_t token)
{
    ReimsVGPUMMIOState *s = ctx;

    return reims_vgpu_dirty_gen(s->dirty, token);
}

static int64_t reims_vgpu_mmio_guest_written_pages(void *ctx, uint64_t token,
                                         uint64_t since_gen, uint64_t *out,
                                         size_t max)
{
    ReimsVGPUMMIOState *s = ctx;

    return reims_vgpu_dirty_written_since(s->dirty, token, since_gen, out, max);
}

static void reims_vgpu_mmio_deliver_actions(ReimsVGPUMMIOState *s);
static void reims_vgpu_mmio_apply_action(ReimsVGPUMMIOState *s, const ReimsVgpuHostAction *a);

static void reims_vgpu_mmio_schedule_bh(void *ctx)
{
    ReimsVGPUMMIOState *s = ctx;

    /* Wakes the drain worker; see drain_thread. Callable from any thread. */
    qemu_mutex_lock(&s->drain_mutex);
    s->drain_pending = true;
    qemu_cond_signal(&s->drain_cond);
    qemu_mutex_unlock(&s->drain_mutex);
}

static void reims_vgpu_mmio_action_bh(void *opaque)
{
    ReimsVGPUMMIOState *s = opaque;
    reims_vgpu_mmio_deliver_actions(s);
}

static void reims_vgpu_mmio_notify_actions(void *ctx)
{
    ReimsVGPUMMIOState *s = ctx;
    if (s && s->action_bh) {
        qemu_bh_schedule(s->action_bh);
    }
}

/*
 * Pop HostActions produced by a prior drain (sync MMIO or BH). Archive paints
 * scanout inside stamp flush; product enqueues ScanoutUpdate — deliver here so
 * logo pixels hit the console without waiting on an idle main loop.
 */
static void reims_vgpu_mmio_deliver_actions(ReimsVGPUMMIOState *s)
{
    ReimsVgpuHostAction action;
    int rc;

    if (s->rust_handle == 0) {
        return;
    }
    while ((rc = reims_vgpu_qemu_device_pop_action(s->rust_handle, &action)) ==
           REIMS_VGPU_QEMU_OK) {
        reims_vgpu_mmio_apply_action(s, &action);
    }
}

/* ---------- Console surface (apple-gfx set_mode / gfx_update) ---------- */

/*
 * Mode change: new DisplaySurface + attach to console.
 * Matches apple-gfx.m set_mode (create + set_surface → cocoa switchSurface).
 * Called only with a finished present's sizeInPixels (CmdDisplaySwap or the
 * first same-geom early paint) — never on a bare size hint without content.
 */
static void reims_vgpu_mmio_set_mode(ReimsVGPUMMIOState *s, uint32_t width,
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
        /* apple-gfx: set_surface alone; cocoa switchSurface resizes the window. */
        qemu_console_set_surface(s->con, s->surface);
    }
    trace_reims_vgpu_mmio_mode_change(width, height);
}

/*
 * Paint a named guest mapping into the QEMU surface, at the geometry the caller
 * was handed:
 *   - that geometry is the guest-presented surface size (PG modeChangeHandler's
 *     sizeInPixels from the named IOSurface — not a host size heuristic), so
 *     set_mode(w,h) is exact, like apple-gfx.m set_mode;
 *   - the copy fills that surface; do not invent or clamp dimensions in C.
 *
 * Returns whether the surface now holds a frame worth showing. Unlike the PCI
 * shim's twin this does not push the console itself — see the newFrame note at
 * the tail, which is a measurement of this pathway and not of that one.
 */
static bool reims_vgpu_mmio_paint_scanout(ReimsVGPUMMIOState *s,
                                          uint32_t mapping_id, uint32_t width,
                                          uint32_t height, uint32_t generation)
{
    uint8_t *dst;
    uint32_t stride;
    int rc;

    /* Zero size is not a present; skip (framework would not mode-change to 0). */
    if (width == 0 || height == 0) {
        return false;
    }
    /*
     * One surface path. There used to be two: reims_vgpu_mmio_set_gpu_mode
     * allocated an alignment-negotiated buffer, attached it as the
     * DisplaySurface backing, and handed it to Rust for a resident-to-buffer
     * GPU copy so no framebuffer bytes crossed the CPU. Its alignment came
     * from VK_EXT_external_memory_host, which is no longer requested, and the
     * buffers had to be retained until teardown because the engine cached
     * their imports. Both the retention and the import are gone.
     */
    reims_vgpu_mmio_set_mode(s, width, height);
    if (!s->surface) {
        return false;
    }

    dst = surface_data(s->surface);
    stride = surface_stride(s->surface);
    rc = reims_vgpu_qemu_scanout_copy(s->rust_handle, mapping_id, dst, stride,
                               width, height, generation);
    if (rc != REIMS_VGPU_QEMU_OK && rc != REIMS_VGPU_QEMU_EMPTY) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: scanout_copy failed mapping=%u %ux%u rc=%d\n",
                      TYPE_REIMS_VGPU_MMIO, mapping_id, width, height, rc);
        return false;
    }
    if (rc == REIMS_VGPU_QEMU_OK) {
        trace_reims_vgpu_mmio_scanout(mapping_id, width, height);
        if (s->con) {
            qemu_console_update_full(s->con);
            s->new_frame_ready = false;
        }
#ifdef CONFIG_ORCHARD_EMBED
        /* One frame into the console, however many rows it touched. */
        orchard_display_note_present();
#endif
    }
    return true;
}

static void reims_vgpu_mmio_apply_scanout(ReimsVGPUMMIOState *s,
                                       const ReimsVgpuHostAction *a)
{
    uint32_t mapping_id = (uint32_t)a->a0;

    if (s->rust_handle == 0 || !s->con) {
        return;
    }
    /*
     * Console ownership, from Rust. This shim used to paint every present it was
     * handed while the PCI shim gated on the same question, so a pre-boundary
     * present naming an unlatched mapping stole the firmware console here and
     * was refused there — one rule, one pathway holding it.
     */
    if (!reims_vgpu_shim_scanout_may_paint(s->rust_handle, mapping_id)) {
        return;
    }
    if (reims_vgpu_mmio_paint_scanout(s, mapping_id, (uint32_t)a->a1,
                                      (uint32_t)a->a2, (uint32_t)a->a3)) {
        s->new_frame_ready = true;
    }
}

static void reims_vgpu_mmio_apply_cursor(ReimsVGPUMMIOState *s,
                                      const ReimsVgpuHostAction *a)
{
    int x = (int)a->a0;
    int y = (int)a->a1;
    bool show = a->a2 != 0;

    if (!s->con) {
        return;
    }
    trace_reims_vgpu_mmio_cursor(a->a0, a->a1, a->a2);
    qemu_console_set_mouse(s->con, x, y, show);
}

/*
 * Pull glyph pixels from Rust and install a QEMUCursor (apple-gfx
 * cursorGlyphHandler role — C only owns the console cursor object).
 */
static void reims_vgpu_mmio_apply_cursor_glyph(ReimsVGPUMMIOState *s)
{
    ReimsVgpuCursorGlyphInfo info;
    QEMUCursor *c;
    g_autofree uint32_t *pixels = NULL;
    int rc;

    if (s->rust_handle == 0 || !s->con) {
        return;
    }
    rc = reims_vgpu_qemu_cursor_glyph_info(s->rust_handle, &info);
    if (rc != REIMS_VGPU_QEMU_OK || info.width == 0 || info.height == 0 ||
        info.pixel_count == 0 ||
        info.pixel_count != info.width * info.height) {
        return;
    }
    pixels = g_new(uint32_t, info.pixel_count);
    rc = reims_vgpu_qemu_cursor_glyph_copy(s->rust_handle, pixels, info.pixel_count);
    if (rc != REIMS_VGPU_QEMU_OK) {
        return;
    }
    c = cursor_alloc(info.width, info.height);
    if (!c) {
        return;
    }
    c->hot_x = info.hot_x;
    c->hot_y = info.hot_y;
    memcpy(c->data, pixels, (size_t)info.pixel_count * sizeof(uint32_t));
    qemu_console_set_cursor(s->con, c);
    cursor_unref(c);
}

static void reims_vgpu_mmio_apply_action(ReimsVGPUMMIOState *s,
                                      const ReimsVgpuHostAction *a)
{
    switch (a->kind) {
    case REIMS_VGPU_HOST_ACTION_IRQ_GFX:
        trace_reims_vgpu_mmio_irq_gfx();
        qemu_irq_pulse(s->irq_gfx);
        break;
    case REIMS_VGPU_HOST_ACTION_IRQ_IOSFC:
        trace_reims_vgpu_mmio_irq_iosfc();
        qemu_irq_pulse(s->irq_iosfc);
        break;
    case REIMS_VGPU_HOST_ACTION_SCANOUT:
        reims_vgpu_mmio_apply_scanout(s, a);
        break;
    case REIMS_VGPU_HOST_ACTION_CURSOR:
        reims_vgpu_mmio_apply_cursor(s, a);
        break;
    case REIMS_VGPU_HOST_ACTION_CURSOR_GLYPH:
        reims_vgpu_mmio_apply_cursor_glyph(s);
        break;
    case REIMS_VGPU_HOST_ACTION_INPUT_KEY:
        reims_vgpu_shim_input_key(s->con, (uint32_t)a->a0, a->a1 != 0);
        break;
    case REIMS_VGPU_HOST_ACTION_INPUT_POINTER_MOVE:
        reims_vgpu_shim_input_pointer_move(s->con, (uint32_t)a->a0,
                                           (uint32_t)a->a1,
                                           (uint32_t)a->a2,
                                           (uint32_t)a->a3);
        break;
    case REIMS_VGPU_HOST_ACTION_INPUT_POINTER_BUTTON:
        reims_vgpu_shim_input_button(s->con, (uint32_t)a->a0, a->a1 != 0);
        break;
    case REIMS_VGPU_HOST_ACTION_WINDOW_CLOSED:
        qemu_system_shutdown_request(SHUTDOWN_CAUSE_HOST_UI);
        break;
    case REIMS_VGPU_HOST_ACTION_TRACE:
    case REIMS_VGPU_HOST_ACTION_NONE:
    default:
        break;
    }
}

static void *reims_vgpu_mmio_drain_thread(void *opaque)
{
    ReimsVGPUMMIOState *s = opaque;

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
                          TYPE_REIMS_VGPU_MMIO, rc);
        }
        /* HostActions are applied on the main loop, under the BQL. */
        qemu_bh_schedule(s->action_bh);
    }
    return NULL;
}

static void reims_vgpu_mmio_drain_stop(ReimsVGPUMMIOState *s)
{
    if (!s->drain_started) {
        return;
    }
    qemu_mutex_lock(&s->drain_mutex);
    s->drain_stopping = true;
    qemu_cond_signal(&s->drain_cond);
    qemu_mutex_unlock(&s->drain_mutex);
    qemu_thread_join(&s->drain_thread);
    s->drain_started = false;
}

static void reims_vgpu_mmio_poll_tick(void *opaque)
{
    ReimsVGPUMMIOState *s = opaque;

    if (s->rust_handle == 0) {
        return;
    }
    if (reims_vgpu_qemu_device_poll(s->rust_handle) == REIMS_VGPU_QEMU_OK) {
        reims_vgpu_mmio_deliver_actions(s);
    }
    timer_mod(s->poll_timer,
              qemu_clock_get_ms(QEMU_CLOCK_HOST) +
              REIMS_VGPU_MMIO_WINDOW_POLL_MS);
}

/*
 * Display refresh (GraphicHwOps.gfx_update).
 *
 * Archive apple-pv-gpu (host/archive/.../apple-pv-gpu.c fb_update_display):
 *   - Pre present-boundary: may re-pull latched front (logo/pill motion).
 *   - Post present-boundary: re-show last painted surface only — never re-read
 *     live guest pages on the Cocoa clock (dual-mid A/B / tile-through).
 *
 * Product tightens the post-boundary push to guest **frame-ready**:
 * CmdDisplaySwap (and early same-geom front paint) set `new_frame_ready` after
 * writing the finished frame into `surface`. Host only calls
 * qemu_console_update_full when that flag is set — not fixed-rate thrash of
 * every vsync with no new guest present (archive present-boundary = newFrame;
 * stamp completes before HostAction apply so guest waiters see stamp first).
 *
 * Always synchronous: every path paints (or declines to) before returning,
 * and nothing calls qemu_console_hw_update_done later, so the answer to
 * GraphicHwOps.gfx_update's "handled synchronously?" is always true.
 */
static bool reims_vgpu_mmio_fb_update(void *opaque)
{
    ReimsVGPUMMIOState *s = opaque;
    uint32_t mapping_id = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t generation = 0;
    uint32_t kind;

    if (!s->con) {
        return true;
    }

    /*
     * Re-drive ONLINE once the enable mask (+0x104 bit 2) is published, so
     * createDisplayAttributes consumes TimingElements. May deliver ScanoutUpdate
     * HostActions (guest present → frame ready).
     */
    if (s->rust_handle != 0 &&
        reims_vgpu_qemu_device_poll(s->rust_handle) == REIMS_VGPU_QEMU_OK) {
        reims_vgpu_mmio_deliver_actions(s);
    }

    /* Host-console ownership is Rust's call; this only paints what it names. */
    kind = reims_vgpu_shim_console_feed(s->rust_handle, &mapping_id, &width,
                                        &height, &generation);

    if (kind == REIMS_VGPU_CONSOLE_FEED_EARLY) {
        /*
         * _EARLY ends at the first DisplaySwap, and Rust guarantees a paintable
         * mapping and geometry with it — so paint at the geometry it named
         * rather than at whatever the current surface happens to be. This shim
         * used to hold the opposite rule ("never resize from the refresh path"),
         * which is a decode rule living in C: `device_console_feed`'s own doc
         * records that both shims re-tested the geometry and that neither
         * should. An _EARLY whose size differed from the surface repainted
         * nothing here, forever, while the PCI shim resized and painted it.
         */
        if (reims_vgpu_mmio_paint_scanout(s, mapping_id, width, height,
                                          generation)) {
            qemu_console_update_full(s->con);
            s->new_frame_ready = false;
        }
        return true;
    }
    if (kind == REIMS_VGPU_CONSOLE_FEED_FIRMWARE) {
        /*
         * The guest is still on its firmware console. On vmapple, mirror the
         * early framebuffer into the QEMU console surface so screendump and
         * UI monitoring capture the boot logo and early rendering.
         */
        if (s->early_fb_ptr && s->early_fb_width && s->early_fb_height) {
            reims_vgpu_mmio_set_mode(s, s->early_fb_width, s->early_fb_height);
            if (s->surface) {
                uint8_t *dst = surface_data(s->surface);
                uint32_t stride = surface_stride(s->surface);
                uint32_t h = s->early_fb_height;
                uint32_t copy_bytes = MIN(stride, s->early_fb_stride);
                for (uint32_t y = 0; y < h; y++) {
                    memcpy(dst + y * stride, s->early_fb_ptr + y * s->early_fb_stride, copy_bytes);
                }
                qemu_console_update_full(s->con);
            }
        }
        return true;
    }

    /* Nothing painted this tick — re-push the last frame if one is pending.
     * _PRODUCT reaches here every tick: apply_scanout does that painting, and
     * this is the hostPresentCount re-show of a guest-finished frame. */
    if (s->new_frame_ready && s->surface) {
        qemu_console_update_full(s->con);
        s->new_frame_ready = false;
    }
    return true;
}

static const GraphicHwOps reims_vgpu_mmio_fb_ops = {
    .gfx_update = reims_vgpu_mmio_fb_update,
};

/* ---------- MMIO (forward only) ---------- */

static uint64_t reims_vgpu_mmio_gfx_read(void *opaque, hwaddr offset,
                                      unsigned size)
{
    ReimsVGPUMMIOState *s = opaque;
    uint64_t val = 0;

    if (s->rust_handle == 0) {
        return 0;
    }
    if (reims_vgpu_qemu_gfx_read(s->rust_handle, offset, size, &val) != REIMS_VGPU_QEMU_OK) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: gfx read failed offset=0x%" HWADDR_PRIx " size=%u\n",
                      TYPE_REIMS_VGPU_MMIO, offset, size);
        return 0;
    }
    trace_reims_vgpu_mmio_gfx_read(offset, val);
    return val;
}

static void reims_vgpu_mmio_gfx_write(void *opaque, hwaddr offset, uint64_t data,
                                   unsigned size)
{
    ReimsVGPUMMIOState *s = opaque;

    if (s->rust_handle == 0) {
        return;
    }
    trace_reims_vgpu_mmio_gfx_write(offset, data);
    /*
     * Before the register write, not after: this is the guest handing the
     * device work, so every guest store ordered before the handoff must be
     * observed before anything that work does can reuse a host-side copy of
     * those pages. Harvesting here is also the only place it can happen — the
     * accelerator's dirty-log sync needs the BQL, which a vCPU MMIO write
     * holds and the drain thread must never take.
     */
    reims_vgpu_dirty_harvest(s->dirty);
    if (reims_vgpu_qemu_gfx_write(s->rust_handle, offset, data, size) != REIMS_VGPU_QEMU_OK) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: gfx write failed offset=0x%" HWADDR_PRIx
                      " data=0x%" PRIx64 " size=%u\n",
                      TYPE_REIMS_VGPU_MMIO, offset, data, size);
        return;
    }
    /* Drain may have enqueued scanout/IRQ under the doorbell MMIO path. */
    reims_vgpu_mmio_deliver_actions(s);
}

static uint64_t reims_vgpu_mmio_iosfc_read(void *opaque, hwaddr offset,
                                        unsigned size)
{
    ReimsVGPUMMIOState *s = opaque;
    uint64_t val = 0;

    if (s->rust_handle == 0) {
        return 0;
    }
    if (reims_vgpu_qemu_iosfc_read(s->rust_handle, offset, size, &val) !=
        REIMS_VGPU_QEMU_OK) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: iosfc read failed offset=0x%" HWADDR_PRIx
                      " size=%u\n",
                      TYPE_REIMS_VGPU_MMIO, offset, size);
        return 0;
    }
    trace_reims_vgpu_mmio_iosfc_read(offset, val);
    return val;
}

static void reims_vgpu_mmio_iosfc_write(void *opaque, hwaddr offset, uint64_t data,
                                     unsigned size)
{
    ReimsVGPUMMIOState *s = opaque;

    if (s->rust_handle == 0) {
        return;
    }
    trace_reims_vgpu_mmio_iosfc_write(offset, data);
    /*
     * The same reason as the gfx path, and both are needed: this shim exposes
     * two guest-facing register windows, and either can be the write that hands
     * the device work. Harvesting on only one would leave the guest-write
     * witness a whole submission stale on whichever rail the guest happens to
     * use. Cheap when nothing has read a generation since the last harvest, so
     * covering both costs a predicate, not a sync.
     */
    reims_vgpu_dirty_harvest(s->dirty);
    if (reims_vgpu_qemu_iosfc_write(s->rust_handle, offset, data, size) !=
        REIMS_VGPU_QEMU_OK) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: iosfc write failed offset=0x%" HWADDR_PRIx
                      " data=0x%" PRIx64 " size=%u\n",
                      TYPE_REIMS_VGPU_MMIO, offset, data, size);
        return;
    }
    reims_vgpu_mmio_deliver_actions(s);
}

static const MemoryRegionOps reims_vgpu_mmio_gfx_ops = {
    .read = reims_vgpu_mmio_gfx_read,
    .write = reims_vgpu_mmio_gfx_write,
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

static const MemoryRegionOps reims_vgpu_mmio_iosfc_ops = {
    .read = reims_vgpu_mmio_iosfc_read,
    .write = reims_vgpu_mmio_iosfc_write,
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
 * Headless input injection, for driving the guest without the host window.
 *
 * On vmapple the guest takes keyboard and pointer through this device's own
 * QemuConsole -- the window thread calls the same three shim entry points --
 * so QMP `input-send-event` reaches nothing: it routes to the default console
 * and needs a device *id*, which a machine-created device does not have.  These
 * properties are that path, addressable by QOM path:
 *
 *   qom-set path=<dev> property=inject-pointer value=(x << 32 | y)
 *   qom-set path=<dev> property=inject-button  value=(code << 1 | down)
 *   qom-set path=<dev> property=inject-key     value=(evdev << 1 | down)
 *
 * Write-only and inert until written, so nothing changes for a normal run.
 */
static void reims_vgpu_mmio_set_inject_pointer(Object *obj, Visitor *v,
                                               const char *name, void *opaque,
                                               Error **errp)
{
    ReimsVGPUMMIOState *s = REIMS_VGPU_MMIO(obj);
    uint64_t value;

    if (!visit_type_uint64(v, name, &value, errp)) {
        return;
    }
    /* The live surface is the coordinate space the window's own moves use. */
    if (s->surface == NULL) {
        error_setg(errp, "no display surface yet");
        return;
    }
    reims_vgpu_shim_input_pointer_move(s->con, (uint32_t)(value >> 32),
                                       (uint32_t)value,
                                       surface_width(s->surface),
                                       surface_height(s->surface));
}

static void reims_vgpu_mmio_set_inject_button(Object *obj, Visitor *v,
                                              const char *name, void *opaque,
                                              Error **errp)
{
    ReimsVGPUMMIOState *s = REIMS_VGPU_MMIO(obj);
    uint64_t value;

    if (!visit_type_uint64(v, name, &value, errp)) {
        return;
    }
    reims_vgpu_shim_input_button(s->con, (uint32_t)(value >> 1),
                                 (value & 1) != 0);
}

static void reims_vgpu_mmio_set_inject_key(Object *obj, Visitor *v,
                                           const char *name, void *opaque,
                                           Error **errp)
{
    ReimsVGPUMMIOState *s = REIMS_VGPU_MMIO(obj);
    uint64_t value;

    if (!visit_type_uint64(v, name, &value, errp)) {
        return;
    }
    reims_vgpu_shim_input_key(s->con, (uint32_t)(value >> 1), (value & 1) != 0);
}

static void reims_vgpu_mmio_init(Object *obj)
{
    ReimsVGPUMMIOState *s = REIMS_VGPU_MMIO(obj);
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);

    /*
     * Same sysbus layout as apple-gfx-mmio: mmio 0/irq 0 = gfx,
     * mmio 1/irq 1 = IOSurface mapper.
     */
    object_property_add(obj, "inject-pointer", "uint64", NULL,
                        reims_vgpu_mmio_set_inject_pointer, NULL, NULL);
    object_property_add(obj, "inject-button", "uint64", NULL,
                        reims_vgpu_mmio_set_inject_button, NULL, NULL);
    object_property_add(obj, "inject-key", "uint64", NULL,
                        reims_vgpu_mmio_set_inject_key, NULL, NULL);

    memory_region_init_io(&s->iomem_gfx, obj, &reims_vgpu_mmio_gfx_ops, s,
                          TYPE_REIMS_VGPU_MMIO ".gfx",
                          REIMS_VGPU_GFX_MMIO_SIZE);
    memory_region_init_io(&s->iomem_iosfc, obj, &reims_vgpu_mmio_iosfc_ops, s,
                          TYPE_REIMS_VGPU_MMIO ".iosfc",
                          REIMS_VGPU_MMIO_IOSFC_MMIO_SIZE);
    sysbus_init_mmio(sbd, &s->iomem_gfx);
    sysbus_init_mmio(sbd, &s->iomem_iosfc);
    sysbus_init_irq(sbd, &s->irq_gfx);
    sysbus_init_irq(sbd, &s->irq_iosfc);

    s->rust_handle = 0;
    s->con = NULL;
    s->surface = NULL;
    s->new_frame_ready = false;
    s->poll_timer = NULL;
    s->action_bh = NULL;
    s->early_fb_ptr = NULL;
    s->early_fb_stride = 0;
    s->early_fb_width = 0;
    s->early_fb_height = 0;
    s->page_views = g_array_new(false, false,
                                sizeof(ReimsVGPUMMIOPageView));
    qemu_mutex_init(&s->page_views_lock);
    memset(&s->host_ops, 0, sizeof(s->host_ops));
}

static bool reims_vgpu_mmio_window_requested(void)
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

static void reims_vgpu_mmio_realize(DeviceState *dev, Error **errp)
{
    ReimsVGPUMMIOState *s = REIMS_VGPU_MMIO(dev);
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
                   TYPE_REIMS_VGPU_MMIO, REIMS_VGPU_QEMU_ABI_VERSION,
                   reims_vgpu_qemu_abi_version());
        return;
    }

    s->action_bh = aio_bh_new(qemu_get_aio_context(), reims_vgpu_mmio_action_bh, s);
    qemu_mutex_init(&s->drain_mutex);
    qemu_cond_init(&s->drain_cond);

    s->host_ops = (ReimsVgpuHostOps){
        .abi_version = REIMS_VGPU_QEMU_ABI_VERSION,
        .struct_size = sizeof(ReimsVgpuHostOps),
        .ctx = s,
        .read_gpa = reims_vgpu_shim_read_gpa,
        .write_gpa = reims_vgpu_shim_write_gpa,
        .mono_ns = reims_vgpu_shim_mono_ns,
        .schedule_bh = reims_vgpu_mmio_schedule_bh,
        .notify_actions = reims_vgpu_mmio_notify_actions,
        .read_kva = reims_vgpu_shim_read_kva,
        .read_xreg = reims_vgpu_mmio_read_xreg,
        .map_pages = reims_vgpu_mmio_map_pages,
        .unmap_pages = reims_vgpu_mmio_unmap_pages,
        .guest_ram_regions = reims_vgpu_shim_guest_ram_regions,
        .is_ram_gpa = reims_vgpu_shim_is_ram_gpa,
        /*
         * 0 on every host.  Darwin releases its mach_vm_remap views eagerly, and
         * on Linux a measured run with this set to 1 -- where the packed-alias
         * path itself refuses, because plain anonymous guest RAM has no fd to
         * MAP_SHARED -- wedged the guest at the login screen: 0 bytes of disk in
         * 30 s with every vCPU spinning.  Setting it needs the alias path to
         * work first, and that needs shared guest RAM, which amdgpu's userptr
         * then refuses for the whole-RAM Vulkan import.
         */
        .map_pages_stable = 0,
        .track_guest_writes = reims_vgpu_mmio_track_guest_writes,
        .untrack_guest_writes = reims_vgpu_mmio_untrack_guest_writes,
        .guest_write_gen = reims_vgpu_mmio_guest_write_gen,
        .guest_written_pages = reims_vgpu_mmio_guest_written_pages,
    };
    s->dirty = reims_vgpu_dirty_new();

    info = (ReimsVgpuQemuCreateInfo){
        .abi_version = REIMS_VGPU_QEMU_ABI_VERSION,
        .struct_size = sizeof(ReimsVgpuQemuCreateInfo),
        .host_ops = &s->host_ops,
        /* arm64e / vmapple guest: 16 KiB pages. */
        .guest_page_shift = REIMS_VGPU_GUEST_PAGE_SHIFT_ARM64E,
    };

    rc = reims_vgpu_qemu_device_create(&info, &out);
    if (rc != REIMS_VGPU_QEMU_OK || out.handle == 0) {
        error_setg(errp, "%s: reims_vgpu_qemu_device_create failed (rc=%d)",
                   TYPE_REIMS_VGPU_MMIO, rc);
        return;
    }
    s->rust_handle = out.handle;
    qemu_thread_create(&s->drain_thread, "reims-vgpu-mmio-drain",
                       reims_vgpu_mmio_drain_thread, s, QEMU_THREAD_JOINABLE);
    s->drain_started = true;

    /*
     * Console only at realize (apple-gfx / archive apple-pv-gpu). Surface size
     * comes from the first ScanoutUpdate (guest-presented geom via Rust), the
     * same role as PGDisplay modeChangeHandler → set_mode. Optional black
     * preferred-mode surface matches archive mode-list EFI boot dims so the
     * cocoa window is not zero-sized before the first present.
     */
    s->con = qemu_graphic_console_create(dev, 0, &reims_vgpu_mmio_fb_ops, s);
    reims_vgpu_mmio_set_mode(s, REIMS_VGPU_EFI_BOOT_WIDTH, REIMS_VGPU_EFI_BOOT_HEIGHT);
    if (s->surface) {
        memset(surface_data(s->surface), 0,
               (size_t)surface_stride(s->surface) * REIMS_VGPU_EFI_BOOT_HEIGHT);
        qemu_console_update_full(s->con);
    }
    /* Hidden software cursor until the guest sends a glyph/show. */
    qemu_console_set_cursor(s->con, cursor_builtin_hidden());
    qemu_console_set_mouse(s->con, 0, 0, false);

    if (reims_vgpu_mmio_window_requested()) {
        rc = reims_vgpu_qemu_window_start(s->rust_handle, REIMS_VGPU_EFI_BOOT_WIDTH,
                                   REIMS_VGPU_EFI_BOOT_HEIGHT);
        if (rc == REIMS_VGPU_QEMU_OK) {
            if (s->early_fb_ptr) {
                reims_vgpu_qemu_window_set_early_fb(s->rust_handle, s->early_fb_ptr,
                                                    s->early_fb_stride,
                                                    s->early_fb_width,
                                                    s->early_fb_height);
            }
#if defined(CONFIG_DARWIN) && !defined(CONFIG_ORCHARD_EMBED)
            reims_vgpu_mmio_window_owner = s;
            qemu_main = reims_vgpu_mmio_window_main_loop;
#endif
        } else {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "%s: host window unavailable rc=%d; using QEMU display\n",
                          TYPE_REIMS_VGPU_MMIO, rc);
        }
    }

    s->poll_timer = timer_new_ms(QEMU_CLOCK_HOST,
                                 reims_vgpu_mmio_poll_tick, s);
    timer_mod(s->poll_timer, qemu_clock_get_ms(QEMU_CLOCK_HOST));

    reims_vgpu_mmio_instance = s;

    backend[0] = '\0';
    reims_vgpu_qemu_backend_name(backend, sizeof(backend));
    trace_reims_vgpu_mmio_realize(s->rust_handle, backend);
}

static void reims_vgpu_mmio_unrealize(DeviceState *dev)
{
    ReimsVGPUMMIOState *s = REIMS_VGPU_MMIO(dev);

    if (reims_vgpu_mmio_instance == s) {
        reims_vgpu_mmio_instance = NULL;
    }
    /* Before the handle goes: the worker drains through it. */
    reims_vgpu_mmio_drain_stop(s);
    if (s->action_bh) {
        qemu_bh_delete(s->action_bh);
        s->action_bh = NULL;
    }
    if (s->poll_timer) {
        timer_del(s->poll_timer);
        timer_free(s->poll_timer);
        s->poll_timer = NULL;
    }
    if (s->rust_handle != 0) {
        reims_vgpu_qemu_window_stop(s->rust_handle);
        reims_vgpu_qemu_device_destroy(s->rust_handle);
        s->rust_handle = 0;
    }
#if defined(CONFIG_DARWIN) && !defined(CONFIG_ORCHARD_EMBED)
    if (reims_vgpu_mmio_window_owner == s) {
        reims_vgpu_mmio_window_owner = NULL;
    }
#endif
    /* After the Rust device is destroyed: no tracked set may outlive the
     * token holder, and the free turns region logging back off. */
    reims_vgpu_dirty_free(s->dirty);
    s->dirty = NULL;
    reims_vgpu_mmio_free_page_views(s);
    g_clear_pointer(&s->page_views, g_array_unref);
    if (s->con) {
        qemu_console_set_surface(s->con, NULL);
    }
    s->surface = NULL;
}

int reims_vgpu_mmio_set_early_fb(const uint8_t *ptr, size_t stride,
                                 uint32_t width, uint32_t height)
{
    if (reims_vgpu_mmio_instance) {
        reims_vgpu_mmio_instance->early_fb_ptr = ptr;
        reims_vgpu_mmio_instance->early_fb_stride = stride;
        reims_vgpu_mmio_instance->early_fb_width = width;
        reims_vgpu_mmio_instance->early_fb_height = height;
        if (reims_vgpu_mmio_instance->rust_handle != 0) {
            return reims_vgpu_qemu_window_set_early_fb(
                reims_vgpu_mmio_instance->rust_handle, ptr, stride, width, height);
        }
        return 0;
    }
    return -1;
}

static void reims_vgpu_mmio_reset(DeviceState *dev)
{
    ReimsVGPUMMIOState *s = REIMS_VGPU_MMIO(dev);

    if (s->rust_handle != 0) {
        reims_vgpu_qemu_device_reset(s->rust_handle);
    }
    reims_vgpu_mmio_free_page_views(s);
    /* Edge-triggered completion IRQs; leave lines deasserted at reset. */
    qemu_set_irq(s->irq_gfx, 0);
    qemu_set_irq(s->irq_iosfc, 0);
    s->new_frame_ready = false;

    /* Restore EFI black surface. */
    reims_vgpu_mmio_set_mode(s, REIMS_VGPU_EFI_BOOT_WIDTH, REIMS_VGPU_EFI_BOOT_HEIGHT);
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

static void reims_vgpu_mmio_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);

    dc->desc = "macOS Paravirtualized GPU (Rust host path)";
    dc->realize = reims_vgpu_mmio_realize;
    dc->unrealize = reims_vgpu_mmio_unrealize;
    device_class_set_legacy_reset(dc, reims_vgpu_mmio_reset);
    /* Created by the vmapple machine (gfx-device property), never -device. */
    dc->user_creatable = false;
    dc->hotpluggable = false;
}

static const TypeInfo reims_vgpu_mmio_types[] = {
    {
        .name = TYPE_REIMS_VGPU_MMIO,
        .parent = TYPE_SYS_BUS_DEVICE,
        .instance_size = sizeof(ReimsVGPUMMIOState),
        .instance_init = reims_vgpu_mmio_init,
        .class_init = reims_vgpu_mmio_class_init,
    },
};

DEFINE_TYPES(reims_vgpu_mmio_types)
