/*
 * The per-CPU TranslationBlock jump cache.
 *
 *  Copyright (c) 2003 Fabrice Bellard
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef ACCEL_TCG_TB_JMP_CACHE_H
#define ACCEL_TCG_TB_JMP_CACHE_H

#include "qemu/rcu.h"
#include "exec/cpu-common.h"

/*
 * 16384 entries rather than 4096. A bigger cache used to cost on every
 * flush, which cleared it entry by entry, and a guest kernel flushes its
 * TLB — and with it this cache — often. The flush is now a generation bump
 * (below), so the size no longer does: on an iPhone running a macOS guest
 * the misses that fell through to the hash table (helper_lookup_tb_ptr,
 * qht_lookup_custom, tb_htable_lookup) were 12-13 % of CPU.
 */
#define TB_JMP_CACHE_BITS 14
#define TB_JMP_CACHE_SIZE (1 << TB_JMP_CACHE_BITS)

/*
 * Invalidated in parallel; all accesses to 'tb' must be atomic.
 * A valid entry is read/written by a single CPU, therefore there is
 * no need for qatomic_rcu_read() and pc is always consistent with a
 * non-NULL value of 'tb'.  Strictly speaking pc is only needed for
 * CF_PCREL, but it's used always for simplicity.
 *
 * An entry is live only while its 'gen' equals the cache's: flushing the
 * whole cache is one increment of 'gen' (tcg_flush_jmp_cache), which retires
 * every entry at once. Entries that are cleared one at a time still clear
 * 'tb', as before.
 */
typedef struct CPUJumpCache {
    struct rcu_head rcu;
    uint32_t gen;
    struct {
        TranslationBlock *tb;
        vaddr pc;
        uint32_t gen;
    } array[TB_JMP_CACHE_SIZE];
} CPUJumpCache;

#endif /* ACCEL_TCG_TB_JMP_CACHE_H */
