/*
 * Guest display and input for an in-process embedder. See
 * include/ui/inferno-embed.h for why this exists.
 *
 * Taken from Inferno-iOS's ui/inferno-embed.c and moved to this tree's console
 * API, which renamed what it calls: register_displaychangelistener is
 * qemu_console_register_listener here, graphic_hw_update is
 * qemu_console_hw_update, and key events by qcode are gone in favour of Linux
 * evdev codes. The listener also follows the first *graphic* console rather
 * than console 0, since here console 0 is whatever was created first.
 *
 * Copyright (c) 2026 Inferno iOS port.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "qemu/osdep.h"
#include "qemu/lockable.h"
#include "qemu/timer.h"
#include "qemu/main-loop.h"
#include "qemu/thread.h"
#include "ui/console.h"
#include "ui/inferno-embed.h"
#include "ui/input.h"
#include "ui/surface.h"
#include "system/system.h"
#include "standard-headers/linux/input-event-codes.h"

/*
 * The app reads frames from its own thread while QEMU redraws them on the main
 * loop, so everything below is behind one mutex. It is deliberately not the big
 * lock: a read must not stop the vCPUs, and taking the big lock here while the
 * main loop waits for this one would be a deadlock waiting to happen. The
 * listener callbacks already run under the big lock, which is why they may take
 * this one, and the reader never takes any lock but this.
 */
typedef struct InfernoDisplay
{
    QemuMutex       lock;
    bool            lock_ready;
    bool            attached;
    DisplaySurface* surface;
    uint32_t        generation;
    /* The union of everything redrawn since the app last looked. */
    bool     dirty;
    uint32_t x0, y0, x1, y1;
    /* Whether the finger is down, so a press is sent only when it changes. */
    uint32_t buttons;
} InfernoDisplay;

static InfernoDisplay             inferno_display;
static DisplayChangeListener      inferno_dcl;

/* Counted under the display's own lock; see InfernoDisplayStats. */
static uint64_t inferno_presents;
static uint64_t inferno_refreshes;
/* Bench rig only: the spacing between frames, to tell a cap from a slow guest. */
static int64_t  inferno_last_present_ns;
static int64_t  inferno_gap_min_ns, inferno_gap_max_ns, inferno_gap_sum_ns;
static uint64_t inferno_gap_count;

static void damage_all_locked(InfernoDisplay* d)
{
    if (d->surface == NULL) {
        d->dirty = false;
        return;
    }
    d->dirty = true;
    d->x0    = 0;
    d->y0    = 0;
    d->x1    = surface_width(d->surface);
    d->y1    = surface_height(d->surface);
}

static void inferno_gfx_switch(DisplayChangeListener* dcl, DisplaySurface* surface)
{
    InfernoDisplay* d = &inferno_display;

    QEMU_LOCK_GUARD(&d->lock);
    d->surface = surface;
    d->generation++;
    damage_all_locked(d);
}

static void inferno_gfx_update(DisplayChangeListener* dcl, int x, int y, int w, int h)
{
    InfernoDisplay* d = &inferno_display;

    if (w <= 0 || h <= 0) { return; }

    QEMU_LOCK_GUARD(&d->lock);
    if (d->surface == NULL) { return; }

    if (!d->dirty) {
        d->dirty = true;
        d->x0    = x;
        d->y0    = y;
        d->x1    = x + w;
        d->y1    = y + h;
        return;
    }
    d->x0 = MIN(d->x0, (uint32_t)x);
    d->y0 = MIN(d->y0, (uint32_t)y);
    d->x1 = MAX(d->x1, (uint32_t)(x + w));
    d->y1 = MAX(d->y1, (uint32_t)(y + h));
}

/*
 * Nothing here draws; this is what drives the machine's own redraw, exactly as
 * a window or a VNC client would.
 */
static void inferno_refresh(DisplayChangeListener* dcl)
{
    InfernoDisplay* d = &inferno_display;

    WITH_QEMU_LOCK_GUARD(&d->lock) { inferno_refreshes++; }
    qemu_console_hw_update(dcl->con);
}

void inferno_display_note_present(void)
{
    InfernoDisplay* d = &inferno_display;

    if (!d->attached) { return; }
    WITH_QEMU_LOCK_GUARD(&d->lock)
    {
        int64_t now = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);

        inferno_presents++;
        if (inferno_last_present_ns != 0) {
            int64_t gap = now - inferno_last_present_ns;

            if (inferno_gap_count == 0 || gap < inferno_gap_min_ns) { inferno_gap_min_ns = gap; }
            if (inferno_gap_count == 0 || gap > inferno_gap_max_ns) { inferno_gap_max_ns = gap; }
            inferno_gap_sum_ns += gap;
            inferno_gap_count++;
        }
        inferno_last_present_ns = now;
    }
}

static void inferno_display_gaps(int64_t* min_ms, int64_t* mean_ms, int64_t* max_ms)
{
    InfernoDisplay* d = &inferno_display;

    QEMU_LOCK_GUARD(&d->lock);
    *min_ms  = inferno_gap_count ? inferno_gap_min_ns / SCALE_MS : 0;
    *max_ms  = inferno_gap_count ? inferno_gap_max_ns / SCALE_MS : 0;
    *mean_ms = inferno_gap_count ? (inferno_gap_sum_ns / (int64_t)inferno_gap_count) / SCALE_MS : 0;
    inferno_gap_count = inferno_gap_sum_ns = 0;
}

void inferno_display_stats(InfernoDisplayStats* out)
{
    InfernoDisplay* d = &inferno_display;

    if (out == NULL) { return; }
    memset(out, 0, sizeof(*out));
    if (!d->attached) { return; }

    QEMU_LOCK_GUARD(&d->lock);
    out->presents     = inferno_presents;
    out->refreshes    = inferno_refreshes;
    inferno_presents  = 0;
    inferno_refreshes = 0;
}

static const DisplayChangeListenerOps inferno_dcl_ops = {
    .dpy_name       = "inferno-embed",
    .dpy_refresh    = inferno_refresh,
    .dpy_gfx_update = inferno_gfx_update,
    .dpy_gfx_switch = inferno_gfx_switch,
};

void inferno_display_attach(void)
{
    InfernoDisplay* d = &inferno_display;
    QemuConsole*    con;

    if (d->attached) { return; }

    /*
     * The first graphic console, not console 0: that is the display device's,
     * and index 0 is only whichever console happened to be created first.
     */
    con = qemu_console_lookup_default();
    if (con == NULL) { return; }

    /* Once for the process: a detach does not destroy it, so a re-attach must not re-init it. */
    if (!d->lock_ready) {
        qemu_mutex_init(&d->lock);
        d->lock_ready = true;
    }
    d->attached = true;

    /* Registering calls gfx_switch with the console's current surface, under d->lock. */
    qemu_console_register_listener(con, &inferno_dcl, &inferno_dcl_ops);

    /*
     * The console already has a surface by now — the machine drew its boot
     * splash into it long before the app got here — so pick it up rather than
     * waiting for a switch that may never come.
     */
    WITH_QEMU_LOCK_GUARD(&d->lock)
    {
        d->surface = qemu_console_surface(con);
        damage_all_locked(d);
    }
}

void inferno_display_detach(void)
{
    InfernoDisplay* d = &inferno_display;

    if (!d->attached) { return; }
    qemu_console_unregister_listener(&inferno_dcl);
    WITH_QEMU_LOCK_GUARD(&d->lock)
    {
        d->surface = NULL;
        d->dirty   = false;
    }
    d->attached = false;
}

void inferno_display_invalidate(void)
{
    InfernoDisplay* d = &inferno_display;

    if (!d->attached) { return; }
    QEMU_LOCK_GUARD(&d->lock);
    damage_all_locked(d);
}

InfernoFrameResult inferno_display_read(void* dst, size_t dst_size, InfernoFrameInfo* info)
{
    InfernoDisplay* d = &inferno_display;
    uint32_t        width, height, x0, y0, x1, y1, row;
    const uint8_t*  src;
    uint8_t*        out;
    int             src_stride;
    size_t          need, span;

    if (info == NULL) { return INFERNO_FRAME_NONE; }
    memset(info, 0, sizeof(*info));
    if (!d->attached) { return INFERNO_FRAME_NONE; }

    QEMU_LOCK_GUARD(&d->lock);
    if (d->surface == NULL) { return INFERNO_FRAME_NONE; }

    width  = surface_width(d->surface);
    height = surface_height(d->surface);

    info->width      = width;
    info->height     = height;
    info->stride     = width * 4;
    info->generation = d->generation;

    need = (size_t)width * height * 4;
    if (dst == NULL || dst_size < need) {
        /* Report the whole screen: the caller is about to start from nothing. */
        info->w = width;
        info->h = height;
        damage_all_locked(d);
        return INFERNO_FRAME_RESIZE;
    }
    if (!d->dirty) { return INFERNO_FRAME_NONE; }

    x0 = MIN(d->x0, width);
    y0 = MIN(d->y0, height);
    x1 = MIN(d->x1, width);
    y1 = MIN(d->y1, height);
    if (x0 >= x1 || y0 >= y1) {
        d->dirty = false;
        return INFERNO_FRAME_NONE;
    }

    src        = surface_data(d->surface);
    src_stride = surface_stride(d->surface);
    out        = dst;
    span       = (size_t)(x1 - x0) * 4;

    for (row = y0; row < y1; row++) {
        memcpy(out + (size_t)row * info->stride + (size_t)x0 * 4, src + (size_t)row * src_stride + (size_t)x0 * 4,
               span);
    }

    info->x  = x0;
    info->y  = y0;
    info->w  = x1 - x0;
    info->h  = y1 - y0;
    d->dirty = false;
    return INFERNO_FRAME_OK;
}

/* ------------------------------------------------------------------ */
/* Input                                                               */
/* ------------------------------------------------------------------ */

void inferno_input_touch(int32_t x, int32_t y, bool pressed)
{
    InfernoDisplay* d = &inferno_display;
    QemuConsole*    con;
    uint32_t        width = 0, height = 0;
    bool            was = false;

    if (!d->attached) { return; }

    WITH_QEMU_LOCK_GUARD(&d->lock)
    {
        if (d->surface == NULL) { return; }
        width  = surface_width(d->surface);
        height = surface_height(d->surface);
        was    = d->buttons != 0;
        d->buttons = pressed ? 1 : 0;
    }
    if (width == 0 || height == 0) { return; }

    x = MAX(0, MIN((int32_t)width - 1, x));
    y = MAX(0, MIN((int32_t)height - 1, y));

    con = inferno_dcl.con;
    bql_lock();
    qemu_input_queue_abs(con, INPUT_AXIS_X, x, 0, width);
    qemu_input_queue_abs(con, INPUT_AXIS_Y, y, 0, height);
    /*
     * Only when it actually changes: a press repeated on every move would read
     * as a click per move instead of a drag.
     */
    if (was != pressed) { qemu_input_queue_btn(con, INPUT_BUTTON_LEFT, pressed); }
    qemu_input_event_sync();
    bql_unlock();
}

void inferno_input_function_key(uint32_t number, bool pressed)
{
    /* Evdev codes: F1..F10 are contiguous, F11 and F12 are not. */
    static const unsigned int keys[] = {
        KEY_F1, KEY_F2, KEY_F3, KEY_F4,  KEY_F5,  KEY_F6,
        KEY_F7, KEY_F8, KEY_F9, KEY_F10, KEY_F11, KEY_F12,
    };

    if (number < 1 || number > ARRAY_SIZE(keys)) { return; }

    bql_lock();
    qemu_input_event_send_key_linux(NULL, keys[number - 1], pressed);
    bql_unlock();
}

/* ------------------------------------------------------------------ */
/* Frame counting with no window, for the bench rig                    */
/* ------------------------------------------------------------------ */

/*
 * The app is what normally drives the machine's redraw: attaching a listener
 * starts the refresh timer, and every frame the guest presents is counted. A
 * headless run has no listener at all, so the display pipe never runs and the
 * frame rate cannot be measured outside the app -- which is exactly what a
 * comparison between two builds of the emulator needs.
 *
 * INFERNO_HEADLESS_FPS=1 attaches the same listener from inside the emulator
 * and reads frames the way the app's pump does, then prints a line a second:
 *
 *   INFERNO-FPS <second> presents=<n> refreshes=<n>
 *
 * Off unless the variable is set, so nothing changes for the app.
 */
static void* inferno_headless_pump(void* arg)
{
    InfernoFrameInfo  info;
    InfernoDisplayStats stats;
    void*             buf  = NULL;
    size_t            size = 0;
    int64_t           next;
    int64_t           gmin = 0, gmean = 0, gmax = 0;
    uint64_t          second = 0;

    next = qemu_clock_get_ms(QEMU_CLOCK_REALTIME) + 1000;
    for (;;) {
        if (inferno_display_read(buf, size, &info) == INFERNO_FRAME_RESIZE) {
            g_free(buf);
            size = (size_t)info.width * info.height * 4;
            buf  = g_malloc0(size);
        }
        g_usleep(1000000 / 60);
        if (qemu_clock_get_ms(QEMU_CLOCK_REALTIME) < next) { continue; }
        next += 1000;
        second++;
        inferno_display_stats(&stats);
        inferno_display_gaps(&gmin, &gmean, &gmax);
        fprintf(stderr,
                "INFERNO-FPS %" PRIu64 " presents=%" PRIu64 " refreshes=%" PRIu64 " gap=%" PRId64 "/%" PRId64 "/%"
                PRId64 "ms\n",
                second, stats.presents, stats.refreshes, gmin, gmean, gmax);
        fflush(stderr);
    }
    return NULL;
}

static void inferno_headless_start(Notifier* n, void* opaque)
{
    static QemuThread thread;

    if (g_strcmp0(getenv("INFERNO_HEADLESS_FPS"), "1") != 0) { return; }

    inferno_display_attach();
    qemu_thread_create(&thread, "inferno.fps", inferno_headless_pump, NULL, QEMU_THREAD_DETACHED);
}

static Notifier inferno_headless_notifier = {.notify = inferno_headless_start};

static void __attribute__((constructor)) inferno_headless_register(void)
{ qemu_add_machine_init_done_notifier(&inferno_headless_notifier); }
