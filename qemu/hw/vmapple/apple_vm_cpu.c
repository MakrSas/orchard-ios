/*
 * Apple Virtual Machine CPU model.
 *
 * A thin subclass of the ARM `max` core for the Orchard `apple-vm` machine.
 * `max` gives us the AArch64 feature set XNU needs (PAC, the v8.x extensions)
 * and — unlike apple-a13/apple-m2 — realizes without any SoC-specific AIC2 /
 * fiq-or wiring, so it drops straight into a GICv3-based virtual machine.
 *
 * This class exists so that Apple-VM hardware quirks (e.g. delivering the
 * architected timer on FIQ the way XNU expects, or shimming IMPDEF system
 * registers) can be added here rather than in the shared ARM core.
 *
 * Copyright (c) 2026 Youssef Elliethy (yaelliethy)
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/module.h"
#include "qom/object.h"
#include "system/tcg.h"
#include "hw/vmapple/apple_vm_cpu.h"
#include "target/arm/cpu.h"
#include "target/arm/cpu-features.h"

static void apple_vm_cpu_initfn(Object *obj)
{
    /*
     * The parent (`max`) instance_init has already enabled the full feature
     * set. Apple-VM-specific instance tweaks go here.
     */
    /* 16 KiB target pages, as the machine asks; see apple_vm_page_bits(). */
    ARM_CPU(obj)->min_page_bits = apple_vm_page_bits();

    /*
     * No BTI. `max` offers it, but no Apple silicon macOS runs on has it, and
     * under TCG it costs on every indirect branch: a BR checks the target
     * page's guard bit through probe_access, a TB entered by BLR/BR runs
     * helper_guarded_page_check, and BTYPE is part of the TB flags, so one
     * function is translated twice. ORCHARD_BTI=1 keeps it, to compare.
     */
    if (g_strcmp0(getenv("ORCHARD_BTI"), "1") != 0) {
        FIELD_DP64_IDREG(&ARM_CPU(obj)->isar, ID_AA64PFR1, BT, 0);
    }

    if (tcg_enabled()) {
        /*
         * PAC is a no-op by default: cheap under TCG, and the kernelcache we
         * boot via -kernel is ChefKiss-patched so its JOP checks never run.
         * A stock kernel (the AVPBooter path loads one off the disk) does run
         * them and dies with "JOP Hash Mismatch Detected (PC, CPSR, or LR
         * corruption)", because no-op signing cannot satisfy a real verify.
         * ORCHARD_REAL_PAUTH=1 selects QEMU's actual PAC implementation.
         */
        const char *real = getenv("ORCHARD_REAL_PAUTH");

        if (real == NULL || g_strcmp0(real, "1") != 0) {
            object_property_set_bool(obj, "pauth-noop", true, NULL);
        } else {
            /*
             * The cheap implementation-defined hash, not architected QARMA5:
             * both are self-consistent, so a guest cannot tell them apart by
             * signing and authenticating alone, and this one is faster.
             */
            object_property_set_bool(obj, "pauth-noop", false, NULL);
            object_property_set_bool(obj, "pauth-impdef", true, NULL);
        }
    }
}

static void apple_vm_cpu_class_init(ObjectClass *oc, const void *data)
{
    /*
     * The parent (`max`) class_init has already installed reset/realize.
     * Apple-VM-specific class overrides (realize wrapper, timer routing) go
     * here.
     */
}

static const TypeInfo apple_vm_cpu_type_info = {
    .name = TYPE_APPLE_VM_CPU,
    .parent = ARM_CPU_TYPE_NAME("max"),
    .instance_init = apple_vm_cpu_initfn,
    .class_init = apple_vm_cpu_class_init,
};

static void apple_vm_cpu_register_types(void)
{
    type_register_static(&apple_vm_cpu_type_info);
}

type_init(apple_vm_cpu_register_types)
