/*
 * reims-vgpu: bus-independent shim services. See reims-vgpu-shim.h.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/timer.h"
#include "hw/core/cpu.h"
#include "exec/cpu-common.h"
#include "system/address-spaces.h"
#include "system/hw_accel.h"
#include "system/memory.h"
#include "ui/console.h"
#include "ui/input.h"
#include "reims_vgpu_qemu_abi.h"
#include "reims-vgpu-shim.h"

uint64_t reims_vgpu_shim_mono_ns(void *ctx)
{
    (void)ctx;
    return (uint64_t)qemu_clock_get_ns(QEMU_CLOCK_HOST);
}

/*
 * RAM-only attrs. `memory = 1` makes the address space reject a translation
 * that resolves to a device rather than RAM (MEMTX_ACCESS_ERROR) instead of
 * performing it. Every GPA reaching read/write_gpa came from a guest-supplied
 * page-entry list, so an entry pointing at one of our own BARs would otherwise
 * re-enter this device's MMIO handler from inside a Rust call that already
 * holds the device lock. Failing closed turns that into a visible decline.
 */
static const MemTxAttrs reims_vgpu_shim_ram_attrs = {
    .memory = true,
};

int reims_vgpu_shim_read_gpa(void *ctx, uint64_t gpa, uint8_t *buf, size_t len)
{
    MemTxResult r;

    (void)ctx;
    if (!buf || len == 0) {
        return 0;
    }
    r = address_space_read(&address_space_memory, gpa,
                           reims_vgpu_shim_ram_attrs, buf, len);
    return r == MEMTX_OK ? 0 : -1;
}

int reims_vgpu_shim_write_gpa(void *ctx, uint64_t gpa, const uint8_t *buf,
                              size_t len)
{
    MemTxResult r;

    (void)ctx;
    if (!buf || len == 0) {
        return 0;
    }
    r = address_space_write(&address_space_memory, gpa,
                            reims_vgpu_shim_ram_attrs, buf, len);
    return r == MEMTX_OK ? 0 : -1;
}

int reims_vgpu_shim_is_ram_gpa(void *ctx, uint64_t gpa)
{
    hwaddr xlat, plen = 1;
    MemoryRegion *mr;
    int ok;

    (void)ctx;
    rcu_read_lock();
    mr = address_space_translate(&address_space_memory, gpa, &xlat, &plen, true,
                                 MEMTXATTRS_UNSPECIFIED);
    ok = mr && memory_region_is_ram(mr) && plen >= 1;
    rcu_read_unlock();
    return ok ? 1 : 0;
}

/*
 * Coalescing state for one guest_ram_regions walk.
 *
 * `found` counts every span the walk produced, including the ones past the
 * caller's `max`. That is the difference between a caller that learns its array
 * was short and one that imports part of the guest's RAM and never finds out.
 */
typedef struct ReimsVgpuRamScan {
    ReimsVgpuGuestRamRegion *out;
    size_t max;
    size_t found;
    bool pending;
    uint64_t gpa;
    uint64_t hva;
    uint64_t len;
} ReimsVgpuRamScan;

static void reims_vgpu_ram_scan_flush(ReimsVgpuRamScan *scan)
{
    if (!scan->pending) {
        return;
    }
    if (scan->found < scan->max) {
        scan->out[scan->found].gpa_base = scan->gpa;
        scan->out[scan->found].host_va = scan->hva;
        scan->out[scan->found].len = scan->len;
    }
    scan->found++;
    scan->pending = false;
}

static bool reims_vgpu_ram_scan_range(Int128 start, Int128 len,
                                      const MemoryRegion *mr,
                                      hwaddr offset_in_region, void *opaque)
{
    ReimsVgpuRamScan *scan = opaque;
    uint64_t gpa, bytes, hva;
    void *ptr;

    /*
     * ROM and ROMD ranges are RAM-backed but the guest cannot store into them,
     * and a ram_device is somebody else's mapping (a VFIO BAR) rather than
     * guest memory. Reporting any of the three would hand the GPU write access
     * to bytes the guest believes are fixed, or does not own at all.
     */
    if (!mr || !memory_region_is_ram(mr) || mr->readonly ||
        memory_region_is_romd(mr) || memory_region_is_ram_device(mr)) {
        reims_vgpu_ram_scan_flush(scan);
        return false;
    }
    /*
     * An Int128 that does not fit a uint64_t cannot be a mapping in this
     * process's address space, so it is not a span anything could import.
     */
    if (int128_gethi(start) != 0 || int128_gethi(len) != 0) {
        reims_vgpu_ram_scan_flush(scan);
        return false;
    }
    gpa = int128_get64(start);
    bytes = int128_get64(len);
    if (bytes == 0) {
        return false;
    }
    ptr = memory_region_get_ram_ptr(mr);
    if (!ptr) {
        reims_vgpu_ram_scan_flush(scan);
        return false;
    }
    hva = (uint64_t)(uintptr_t)ptr + (uint64_t)offset_in_region;

    /*
     * Merge only when the next range continues this one in *both* address
     * spaces. A board that splits one RAMBlock across several flat ranges then
     * costs one span, while two blocks that abut in GPA but not in host VA stay
     * separate — merging those would make an offset inside the span name the
     * wrong host bytes, which is exactly the failure importing a RAMBlock at a
     * time exists to avoid.
     */
    if (scan->pending && scan->gpa + scan->len == gpa &&
        scan->hva + scan->len == hva && scan->len <= UINT64_MAX - bytes) {
        scan->len += bytes;
        return false;
    }
    reims_vgpu_ram_scan_flush(scan);
    scan->pending = true;
    scan->gpa = gpa;
    scan->hva = hva;
    scan->len = bytes;
    return false;
}

int reims_vgpu_shim_guest_ram_regions(void *ctx, ReimsVgpuGuestRamRegion *out,
                                      size_t max)
{
    ReimsVgpuRamScan scan = { 0 };

    (void)ctx;
    if (!out || max == 0) {
        return REIMS_VGPU_GUEST_RAM_ERR_ARGS;
    }
    scan.out = out;
    scan.max = max;

    rcu_read_lock();
    flatview_for_each_range(address_space_to_flatview(&address_space_memory),
                            reims_vgpu_ram_scan_range, &scan);
    rcu_read_unlock();
    reims_vgpu_ram_scan_flush(&scan);

    if (scan.found == 0) {
        return REIMS_VGPU_GUEST_RAM_ERR_NO_RAM;
    }
    /*
     * A span count that does not fit the return type is not a machine anyone
     * runs. Reporting it as "no RAM" would be a lie, so it is the argument
     * refusal: the caller asked for an answer this return value cannot carry.
     */
    if (scan.found > (size_t)INT_MAX) {
        return REIMS_VGPU_GUEST_RAM_ERR_ARGS;
    }
    return (int)scan.found;
}

int reims_vgpu_shim_read_kva(void *ctx, uint64_t kva, uint8_t *buf, size_t len)
{
    CPUState *cs = current_cpu;

    (void)ctx;
    if (!buf || len == 0) {
        return 0;
    }
    if (!cs) {
        return -2;
    }
    cpu_synchronize_state(cs);
    return cpu_memory_rw_debug(cs, kva, buf, len, false) == 0 ? 0 : -1;
}

static InputButton reims_vgpu_shim_button(uint32_t code, bool *ok)
{
    *ok = true;
    switch (code) {
    case REIMS_VGPU_BUTTON_LEFT:
        return INPUT_BUTTON_LEFT;
    case REIMS_VGPU_BUTTON_MIDDLE:
        return INPUT_BUTTON_MIDDLE;
    case REIMS_VGPU_BUTTON_RIGHT:
        return INPUT_BUTTON_RIGHT;
    case REIMS_VGPU_BUTTON_WHEEL_UP:
        return INPUT_BUTTON_WHEEL_UP;
    case REIMS_VGPU_BUTTON_WHEEL_DOWN:
        return INPUT_BUTTON_WHEEL_DOWN;
    case REIMS_VGPU_BUTTON_SIDE:
        return INPUT_BUTTON_SIDE;
    case REIMS_VGPU_BUTTON_EXTRA:
        return INPUT_BUTTON_EXTRA;
    case REIMS_VGPU_BUTTON_WHEEL_LEFT:
        return INPUT_BUTTON_WHEEL_LEFT;
    case REIMS_VGPU_BUTTON_WHEEL_RIGHT:
        return INPUT_BUTTON_WHEEL_RIGHT;
    default:
        *ok = false;
        return INPUT_BUTTON_LEFT;
    }
}

bool reims_vgpu_shim_scanout_may_paint(uint64_t rust_handle, uint32_t mapping_id)
{
    uint32_t may = 0;

    if (rust_handle == 0) {
        return false;
    }
    if (reims_vgpu_qemu_scanout_may_paint(rust_handle, mapping_id, &may) !=
        REIMS_VGPU_QEMU_OK) {
        return false;
    }
    return may != 0;
}

uint32_t reims_vgpu_shim_console_feed(uint64_t rust_handle, uint32_t *out_mid,
                                      uint32_t *out_w, uint32_t *out_h,
                                      uint32_t *out_gen)
{
    uint32_t kind = REIMS_VGPU_CONSOLE_FEED_FIRMWARE;

    if (rust_handle == 0) {
        return REIMS_VGPU_CONSOLE_FEED_FIRMWARE;
    }
    if (reims_vgpu_qemu_console_feed(rust_handle, &kind, out_mid, out_w, out_h,
                                     out_gen) != REIMS_VGPU_QEMU_OK) {
        return REIMS_VGPU_CONSOLE_FEED_FIRMWARE;
    }
    return kind;
}

void reims_vgpu_shim_input_key(QemuConsole *con, uint32_t evdev, bool down)
{
    if (!con) {
        return;
    }
    /* Takes the evdev code and does the qcode translation itself. */
    qemu_input_event_send_key_linux(con, evdev, down);
}

void reims_vgpu_shim_input_pointer_move(QemuConsole *con, uint32_t x,
                                        uint32_t y, uint32_t w, uint32_t h)
{
    if (!con || w == 0 || h == 0) {
        return;
    }
    qemu_input_queue_abs(con, INPUT_AXIS_X, (int)x, 0, (int)w);
    qemu_input_queue_abs(con, INPUT_AXIS_Y, (int)y, 0, (int)h);
    qemu_input_event_sync();
}

void reims_vgpu_shim_input_button(QemuConsole *con, uint32_t code, bool down)
{
    InputButton button;
    bool ok;

    if (!con) {
        return;
    }
    button = reims_vgpu_shim_button(code, &ok);
    if (!ok) {
        return;
    }
    qemu_input_queue_btn(con, button, down);
    qemu_input_event_sync();
}
