/*
 * reims-vgpu: guest-write tracking over the hypervisor dirty bitmap.
 *
 * A paravirtual surface's storage is plain guest RAM. The guest CPU stores
 * into it with no device operation, so nothing the Rust device counts can
 * witness such a store, and every host-side copy of those pages (an engine
 * resident, a cache entry) is stale from that instant with nothing to say so.
 * The hypervisor's dirty bitmap is the only witness. This is the device-side
 * adapter for it, shared by both shims.
 *
 * The Rust side registers a page set once per mapping incarnation and then
 * reads a *generation*. A generation rather than a consume-on-read flag
 * because a consumed flag is correct for exactly one reader: the first draw of
 * a frame would eat the report and every later draw would be told the surface
 * is clean.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef REIMS_VGPU_DIRTY_H
#define REIMS_VGPU_DIRTY_H

#include <stdint.h>
#include <stddef.h>

typedef struct ReimsVgpuDirty ReimsVgpuDirty;

/* Create/destroy. Destroy runs with the BQL held: it turns region logging
 * back off, which is a MemoryRegion transaction. */
ReimsVgpuDirty *reims_vgpu_dirty_new(void);
void reims_vgpu_dirty_free(ReimsVgpuDirty *d);

/*
 * Register `count` page-aligned GPAs of `page_size` bytes as one tracked set.
 * Returns a non-zero opaque token, or 0 when the set cannot be tracked.
 * Safe from any thread.
 *
 * The token's generation reads back as 0 ("cannot tell") until one harvest has
 * passed, and until two when that harvest was the one that turned dirty logging
 * on for the pages' regions: writes older than the enable were never recorded
 * and must not be mistaken for their absence. Guest RAM is one MemoryRegion, so
 * after the first tracked surface of a boot no harvest enables anything and the
 * window is one.
 */
uint64_t reims_vgpu_dirty_track(ReimsVgpuDirty *d, const uint64_t *gpas,
                                size_t count, size_t page_size);

/* Release a token. Safe from any thread; a stale or zero token is ignored. */
void reims_vgpu_dirty_untrack(ReimsVgpuDirty *d, uint64_t token);

/*
 * Monotonic count of harvests that saw some page of the set written, or 0 for
 * an unknown or not-yet-armed token. Safe from any thread, and idempotent:
 * two reads returning the same value prove no harvest observed a write in
 * between.
 */
uint64_t reims_vgpu_dirty_gen(ReimsVgpuDirty *d, uint64_t token);

/*
 * Which pages of the set were written, not just whether any were.
 *
 * Fills `out` with the page-aligned GPAs of `token`'s set whose most recent
 * observed write is newer than `since_gen`, and returns how many. `since_gen`
 * is a value a caller previously read from reims_vgpu_dirty_gen() and recorded
 * next to a host-side copy of the pages.
 *
 * Returns -1 for every case where the answer is not knowable and the caller must
 * assume the whole set was written: an unknown token, a token whose generation
 * is still unreadable, a `since_gen` of 0 (no recorded observation to compare
 * against), or a set with more written pages than `max` can hold. A truncated
 * list would say "these pages and no others", which is the one answer that turns
 * a conservative caller into a wrong one.
 *
 * Safe from any thread.
 */
int64_t reims_vgpu_dirty_written_since(ReimsVgpuDirty *d, uint64_t token,
                                       uint64_t since_gen, uint64_t *out,
                                       size_t max);

/*
 * Fold the hypervisor bitmap into every tracked set. **BQL thread only** —
 * this enables MemoryRegion dirty logging and drives the accelerator's
 * log_sync, neither of which is safe off the main thread.
 *
 * Call it where the guest hands the device work, so that every guest store
 * ordered before that handoff is observed before the work runs. Cheap when
 * nothing is tracked or when no generation has been read since the last call.
 */
void reims_vgpu_dirty_harvest(ReimsVgpuDirty *d);

#endif /* REIMS_VGPU_DIRTY_H */
