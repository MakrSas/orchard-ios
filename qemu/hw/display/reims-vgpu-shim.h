/*
 * reims-vgpu: shim services that do not depend on which bus attached the
 * device, shared by reims-vgpu-pci and reims-vgpu-mmio.
 *
 * A service belongs here when its body reads nothing from the device state:
 * either it ignores the HostOps `ctx` entirely (host clock, address-space
 * queries, vCPU-relative reads) or it needs only the console, which the
 * caller passes explicitly. Anything that touches the bus object, the
 * per-device dirty tracker, or a bus-specific trace event stays in its shim.
 *
 * The point is not line count. A duplicated button table or address-space
 * predicate is a table that can drift, and a drift between the two shims is
 * a bug the guest sees on exactly one pathway.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef REIMS_VGPU_SHIM_H
#define REIMS_VGPU_SHIM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "qemu/typedefs.h"
/* ReimsVgpuGuestRamRegion, for the span enumeration below. */
#include "reims_vgpu_qemu_abi.h"

/*
 * The five below match HostOps function-pointer signatures, so both shims'
 * host_ops tables point at them directly; `ctx` is accepted and ignored.
 */

/* Host monotonic clock in nanoseconds. */
uint64_t reims_vgpu_shim_mono_ns(void *ctx);

/*
 * Guest-physical byte access. Both use RAM-only transaction attrs, so a GPA
 * that resolves to a device is rejected rather than performed — a guest page
 * entry pointing at one of our own BARs must not re-enter device MMIO from
 * inside a Rust call. Return 0 on success (including an empty request), -1 on
 * a failed transaction.
 */
int reims_vgpu_shim_read_gpa(void *ctx, uint64_t gpa, uint8_t *buf, size_t len);
int reims_vgpu_shim_write_gpa(void *ctx, uint64_t gpa, const uint8_t *buf,
                              size_t len);

/* 1 = guest RAM, 0 = not (MMIO / ROM / unmapped). Mapper page-entry accept. */
int reims_vgpu_shim_is_ram_gpa(void *ctx, uint64_t gpa);

/*
 * Where guest RAM lives in this process, as stable (gpa_base, host_va, len)
 * spans. See the ReimsVgpuHostOps.guest_ram_regions comment in
 * reims_vgpu_qemu_abi.h for the contract; this is the one implementation of it
 * and both shims point their table entry straight at it.
 *
 * Bus-independent by construction: it reads the system address space and
 * nothing from the device. Writing it once is not a line-count argument — the
 * two shims run on different hosts, so a second copy is a divergence that shows
 * up on exactly the pathway nobody is booting.
 */
int reims_vgpu_shim_guest_ram_regions(void *ctx, ReimsVgpuGuestRamRegion *out,
                                      size_t max);

/*
 * Guest kernel VA -> host buffer, for MappingInternal / page-table walks.
 *
 * MUST run on a vCPU thread with current_cpu set (typically the iosfc
 * producer MMIO path). Never falls back to first_cpu: from the drain BH
 * that would do_run_on_cpu while the vCPU may be blocked in MMIO waiting
 * for the Rust DEVICES mutex — classic AB-BA hang (UI "not responding").
 * Returns 0 on success, -2 with no current vCPU, -1 on a failed read.
 */
int reims_vgpu_shim_read_kva(void *ctx, uint64_t kva, uint8_t *buf, size_t len);

/*
 * May a present naming `mapping_id` paint the host console right now?
 *
 * The verdict comes from Rust whole; this only forwards it, and answers false
 * for a device that has no handle or cannot answer. Do NOT rebuild it from
 * `reims_vgpu_shim_console_feed`'s kind and mapping id — that reconstruction is
 * what the two shims had drifted on, x86 gating and arm64 painting every
 * present it was handed.
 */
bool reims_vgpu_shim_scanout_may_paint(uint64_t rust_handle, uint32_t mapping_id);

/*
 * Who owns the host console right now: a `REIMS_VGPU_CONSOLE_FEED_*`. The
 * `out_*` are filled only for _EARLY, and are left untouched otherwise.
 *
 * A device that has no handle, or whose call does not return OK, is reported as
 * _FIRMWARE. That is the answer, not a fallback: the pre-boundary firmware
 * console is what owns the screen until Rust says otherwise, so failing closed
 * is the only reading that cannot paint product pixels over it. The arm64 shim
 * used to conflate "the call failed" with "not _EARLY" and fall through to its
 * post-boundary re-push, which is the shim inventing a policy for "no answer" —
 * `device_console_feed`'s own doc names that as the thing being removed.
 */
uint32_t reims_vgpu_shim_console_feed(uint64_t rust_handle, uint32_t *out_mid,
                                      uint32_t *out_w, uint32_t *out_h,
                                      uint32_t *out_gen);

/*
 * Host-owned-window input: replay a neutral Rust input action through the
 * QEMU input subsystem. Input routes to the guest's active handlers
 * (usb-kbd / usb-tablet) independent of any display, so these work with
 * -display none. All business logic (platform key -> evdev, scroll delta ->
 * wheel notches, coordinate origin) lives in Rust; the shim only translates
 * the neutral wire form into the QEMU input ABI, which owns the QEMU-side
 * keycode and button enums. A NULL console drops the event.
 */

/* `evdev` is a Linux evdev code; QEMU drops codes it cannot map to a qcode,
 * so no shim-side keycode table is needed. */
void reims_vgpu_shim_input_key(QemuConsole *con, uint32_t evdev, bool down);

/* Absolute pointer (usb-tablet): scales window pixels into the abs range. */
void reims_vgpu_shim_input_pointer_move(QemuConsole *con, uint32_t x,
                                        uint32_t y, uint32_t w, uint32_t h);

/* `code` is a REIMS_VGPU_BUTTON_*; an unrecognised one drops the event. */
void reims_vgpu_shim_input_button(QemuConsole *con, uint32_t code, bool down);

/* Early boot framebuffer registration for MMIO platform (vmapple / sysbus). */
int reims_vgpu_mmio_set_early_fb(const uint8_t *ptr, size_t stride,
                                 uint32_t width, uint32_t height);

#endif /* REIMS_VGPU_SHIM_H */
