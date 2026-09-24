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
 * 4096 entries, as upstream. A generation-based flush (below) made a bigger
 * cache free to empty, but 16384 entries measured no better on an iPhone
 * running a macOS guest: the lookups that remain are not misses, and a
 * 384 KiB cache per vCPU no longer fits in the core's L1.
 */
#define TB_JMP_CACHE_BITS 12
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
    /*
     * Counters for the app's profile line (orchard_tcg_stats). Each is
     * written by its own vCPU only, except 'flushes', which other vCPUs
     * bump atomically.
     */
    uint64_t lookups, misses, translations, exceptions, flushes;
    struct {
        TranslationBlock *tb;
        vaddr pc;
        uint32_t gen;
    } array[TB_JMP_CACHE_SIZE];
} CPUJumpCache;

#endif /* ACCEL_TCG_TB_JMP_CACHE_H */
