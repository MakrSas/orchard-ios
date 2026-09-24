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
 * 16384 entries, four times upstream's. On a macOS guest some 12% of the
 * 20-odd million lookups a second missed, nearly all because the slot held
 * another pc: with upstream's layout every 16 KiB page shares 64 slots, and
 * a hot kernel page alone has more TBs than that. Each miss costs a trip
 * through the QHT, far more than a jump-cache line in L2 instead of L1.
 * Page flushes and the user-half flush stay cheap (tb-hash.h, below).
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
 * An entry is live only while its 'gen' equals the cache's generation for
 * its half of the address space: 'gen' for pc with bit 63 set (the kernel's
 * half), 'gen_lo' for the rest. Flushing the whole cache is one increment of
 * 'gen' and of the epoch in 'gen_lo' (tcg_flush_jmp_cache), which retires
 * every entry at once. Entries that are cleared one at a time still clear
 * 'tb', as before.
 *
 * 'gen_lo' is an epoch (high 16 bits) and the current ASID (low 16): user
 * TBs are tagged with the process they ran in, as a real TLB tags its
 * entries, so switching processes and back finds them still there
 * (tb_jmp_cache_set_asid). The guest has to invalidate by TLBI whatever
 * mapping it changes under an ASID, and every TLB flush clears the jump
 * cache entries at the flushed addresses, of all ASIDs, or bumps the epoch.
 */
typedef struct CPUJumpCache {
    struct rcu_head rcu;
    uint32_t gen, gen_lo;
    /*
     * Counters for the app's profile line (orchard_tcg_stats). Each is
     * written by its own vCPU only, except 'flushes', which other vCPUs
     * bump atomically.
     */
    uint64_t lookups, misses, translations, exceptions, flushes;
    /* Why lookups missed: an emptied slot, another pc's, this pc's other TB */
    uint64_t miss_empty, miss_other_pc, miss_same_pc;
    struct {
        TranslationBlock *tb;
        vaddr pc;
        uint32_t gen;
    } array[TB_JMP_CACHE_SIZE];
} CPUJumpCache;

/* Retire every user-half entry, of all ASIDs. True when the epoch wrapped. */
static inline bool tb_jmp_cache_bump_lo(CPUJumpCache *jc)
{
    uint32_t old = qatomic_read(&jc->gen_lo), seen, new;

    for (;;) {
        new = old + 0x10000;
        seen = qatomic_cmpxchg(&jc->gen_lo, old, new);
        if (seen == old) {
            return (new >> 16) == 0;
        }
        old = seen;
    }
}

/* Make @asid's user-half entries the live ones. */
static inline void tb_jmp_cache_set_asid(CPUJumpCache *jc, uint16_t asid)
{
    uint32_t old = qatomic_read(&jc->gen_lo), seen;

    while ((seen = qatomic_cmpxchg(&jc->gen_lo, old,
                                   (old & 0xffff0000u) | asid)) != old) {
        old = seen;
    }
}

/* The generation an entry for @pc must carry to be live. */
static inline uint32_t tb_jmp_cache_gen(CPUJumpCache *jc, vaddr pc)
{
    return (int64_t)pc < 0 ? qatomic_read(&jc->gen) : qatomic_read(&jc->gen_lo);
}

#endif /* ACCEL_TCG_TB_JMP_CACHE_H */
