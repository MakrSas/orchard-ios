/*
 * Apple Virtual Machine CPU — a custom CPU model for the Inferno `apple-vm`
 * machine. Subclasses the ARM `max` core (which realizes standalone, with no
 * AIC2/fiq-or plumbing that the apple-a13/apple-m2 SoC cores demand), giving us
 * a CPU class we fully own — a place to hook Apple-VM-specific behaviour
 * (timer routing to FIQ, IMPDEF sysregs XNU-vmapple pokes, etc.) without
 * touching the shared ARM cores.
 *
 * Copyright (c) 2026 Youssef Elliethy (yaelliethy)
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef HW_VMAPPLE_APPLE_VM_CPU_H
#define HW_VMAPPLE_APPLE_VM_CPU_H

#include "target/arm/cpu-qom.h"

/* Resolves to "apple-vm-arm-cpu"; usable as `-cpu apple-vm` too. */
#define TYPE_APPLE_VM_CPU ARM_CPU_TYPE_NAME("apple-vm")

/*
 * The smallest page the apple-vm machine and its CPU ask QEMU for, as a
 * shift: 14, 16 KiB, the only granule an Apple-silicon macOS kernel uses.
 *
 * QEMU's softmmu TLB holds one entry per target page. At its 4 KiB default a
 * 16 KiB guest page takes four entries, four misses and, for each, a walk of
 * the guest's tables; on an iPhone the address translation around those
 * misses was 10-15 % of all CPU time. A mapping smaller than the target page
 * still works — QEMU refills it on every access instead of caching it — so
 * a 4 KiB mapping, should any code make one, is slow rather than wrong.
 * ORCHARD_TARGET_PAGE_BITS=12 restores 4 KiB, to compare or to rule it out.
 */
static inline int apple_vm_page_bits(void)
{
    const char *bits = getenv("ORCHARD_TARGET_PAGE_BITS");

    return bits && strcmp(bits, "12") == 0 ? 12 : 14;
}

#endif /* HW_VMAPPLE_APPLE_VM_CPU_H */
