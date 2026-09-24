/*
 * internal execution defines for qemu
 *
 *  Copyright (c) 2003 Fabrice Bellard
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */

#ifndef EXEC_TB_HASH_H
#define EXEC_TB_HASH_H

#include "exec/vaddr.h"
#include "exec/target_page.h"
#include "exec/translation-block.h"
#include "qemu/xxhash.h"
#include "tb-jmp-cache.h"

#ifdef CONFIG_SOFTMMU

/*
 * The cache is 32 groups of 512 slots. A page's TBs all live in one group,
 * picked by a multiplicative hash of the page number, so a TLB flush of a
 * page looks at 512 slots and empties only those whose pc is on that page
 * (tb_jmp_cache_clear_page). Upstream had 64 groups of 64, took the group
 * from a few page bits, and emptied the whole group on every page flush.
 */
#define TB_JMP_PAGE_BITS 9
#define TB_JMP_PAGE_SIZE (1 << TB_JMP_PAGE_BITS)
#define TB_JMP_ADDR_MASK (TB_JMP_PAGE_SIZE - 1)
#define TB_JMP_PAGE_MASK (TB_JMP_CACHE_SIZE - TB_JMP_PAGE_SIZE)

static inline unsigned int tb_jmp_cache_hash_page(vaddr pc)
{
    uint64_t page = pc >> TARGET_PAGE_BITS;

    return ((page * 0x9e3779b97f4a7c15ull)
            >> (64 - (TB_JMP_CACHE_BITS - TB_JMP_PAGE_BITS)))
           << TB_JMP_PAGE_BITS;
}

/*
 * The slot within the group folds in every bit of the offset, and the TB
 * flags too: a kernel routine run both with and without PAN (copyin,
 * copyout) is two TBs at one pc, and they do not take turns evicting each
 * other.
 */
static inline unsigned int tb_jmp_cache_hash_func(vaddr pc, uint32_t flags,
                                                  uint64_t cs_base)
{
    uint64_t f = (flags ^ cs_base) * 0x9e3779b97f4a7c15ull;

    return tb_jmp_cache_hash_page(pc) |
           (((pc >> 2) ^ (pc >> 11) ^ (f >> 55)) & TB_JMP_ADDR_MASK);
}

#else

/* In user-mode we can get better hashing because we do not have a TLB */
static inline unsigned int tb_jmp_cache_hash_func(vaddr pc, uint32_t flags,
                                                  uint64_t cs_base)
{
    return (pc ^ (pc >> TB_JMP_CACHE_BITS)) & (TB_JMP_CACHE_SIZE - 1);
}

#endif /* CONFIG_SOFTMMU */

static inline
uint32_t tb_hash_func(tb_page_addr_t phys_pc, vaddr pc,
                      uint32_t flags, uint64_t flags2, uint32_t cf_mask)
{
    return qemu_xxhash8(phys_pc, pc, flags2, flags, cf_mask);
}

#endif
