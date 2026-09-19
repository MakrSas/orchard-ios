/*
 * ASPEED ACRY Engine
 *
 * Copyright (C) 2026 ASPEED Technology Inc.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef ASPEED_ACRY_H
#define ASPEED_ACRY_H

#include "hw/core/sysbus.h"
#include "system/memory.h"

#define TYPE_ASPEED_ACRY "aspeed.acry"
OBJECT_DECLARE_SIMPLE_TYPE(AspeedACRYState, ASPEED_ACRY)

#define ASPEED_ACRY_NR_REGS     (0x400 >> 2)
/* Max size of the "data" (message) field within the SRAM buffer. */
#define ASPEED_ACRY_DATA_MAX_LEN 0x800
#define ASPEED_ACRY_MAX_BITS     4096
/* Max exponent/modulus size for a 4096-bit RSA key, in bytes. */
#define ASPEED_ACRY_MAX_BYTES    (ASPEED_ACRY_MAX_BITS / 8)

struct AspeedACRYState {
    SysBusDevice parent_obj;

    MemoryRegion iomem;
    qemu_irq irq;

    uint32_t regs[ASPEED_ACRY_NR_REGS];

    MemoryRegion *dram_mr;
    MemoryRegion *sram_mr;
    AddressSpace dram_as;
    AddressSpace sram_as;
};

#endif /* ASPEED_ACRY_H */
