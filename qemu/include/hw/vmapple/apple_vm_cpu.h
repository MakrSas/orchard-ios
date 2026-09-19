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

#endif /* HW_VMAPPLE_APPLE_VM_CPU_H */
