/*
 * Guest display and input for an in-process embedder. See
 * include/ui/orchard-embed.h for why this exists.
 *
 * Taken from Inferno-iOS's ui/inferno-embed.c and moved to this tree's console
 * API, which renamed what it calls: register_displaychangelistener is
 * qemu_console_register_listener here, graphic_hw_update is
 * qemu_console_hw_update, and key events by qcode are gone in favour of Linux
 * evdev codes. The listener also follows the first *graphic* console rather
 * than console 0, since here console 0 is whatever was created first.
 *
 * Copyright (c) 2026 Makr (Inferno-iOS, Orchard iOS port).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/lockable.h"
#include "qemu/timer.h"
#include "qemu/main-loop.h"
#include "qemu/thread.h"
#include "ui/console.h"
#include "ui/orchard-embed.h"
#include "ui/input.h"
#include "ui/surface.h"
#include "system/system.h"
#include "block/block.h"
#include "standard-headers/linux/input-event-codes.h"

/*
 * The app reads frames from its own thread while QEMU redraws them on the main
 * loop, so everything below is behind one mutex. It is deliberately not the big
 * lock: a read must not stop the vCPUs, and taking the big lock here while the
 * main loop waits for this one would be a deadlock waiting to happen. The
 * listener callbacks already run under the big lock, which is why they may take
 * this one, and the reader never takes any lock but this.
 */
typedef struct OrchardDisplay
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
} OrchardDisplay;

static OrchardDisplay             orchard_display;
static DisplayChangeListener      orchard_dcl;

/* Counted under the display's own lock; see OrchardDisplayStats. */
static uint64_t orchard_presents;
static uint64_t orchard_refreshes;
/* Bench rig only: the spacing between frames, to tell a cap from a slow guest. */
static int64_t  orchard_last_present_ns;
static int64_t  orchard_gap_min_ns, orchard_gap_max_ns, orchard_gap_sum_ns;
static uint64_t orchard_gap_count;

static void damage_all_locked(OrchardDisplay* d)
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

static void orchard_gfx_switch(DisplayChangeListener* dcl, DisplaySurface* surface)
{
    OrchardDisplay* d = &orchard_display;

    QEMU_LOCK_GUARD(&d->lock);
    d->surface = surface;
    d->generation++;
    damage_all_locked(d);
}

static void orchard_gfx_update(DisplayChangeListener* dcl, int x, int y, int w, int h)
{
    OrchardDisplay* d = &orchard_display;

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
static void orchard_refresh(DisplayChangeListener* dcl)
{
    OrchardDisplay* d = &orchard_display;

    WITH_QEMU_LOCK_GUARD(&d->lock) { orchard_refreshes++; }
    qemu_console_hw_update(dcl->con);
}

void orchard_display_note_present(void)
{
    OrchardDisplay* d = &orchard_display;

    if (!d->attached) { return; }
    WITH_QEMU_LOCK_GUARD(&d->lock)
    {
        int64_t now = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);

        orchard_presents++;
        if (orchard_last_present_ns != 0) {
            int64_t gap = now - orchard_last_present_ns;

            if (orchard_gap_count == 0 || gap < orchard_gap_min_ns) { orchard_gap_min_ns = gap; }
            if (orchard_gap_count == 0 || gap > orchard_gap_max_ns) { orchard_gap_max_ns = gap; }
            orchard_gap_sum_ns += gap;
            orchard_gap_count++;
        }
        orchard_last_present_ns = now;
    }
}

static void orchard_display_gaps(int64_t* min_ms, int64_t* mean_ms, int64_t* max_ms)
{
    OrchardDisplay* d = &orchard_display;

    QEMU_LOCK_GUARD(&d->lock);
    *min_ms  = orchard_gap_count ? orchard_gap_min_ns / SCALE_MS : 0;
    *max_ms  = orchard_gap_count ? orchard_gap_max_ns / SCALE_MS : 0;
    *mean_ms = orchard_gap_count ? (orchard_gap_sum_ns / (int64_t)orchard_gap_count) / SCALE_MS : 0;
    orchard_gap_count = orchard_gap_sum_ns = 0;
}

/*
 * Write everything the block layer holds for the guest's disks to the files:
 * qcow2's cached tables, then fsync. For the app going to the background,
 * where iOS may end it without another word. Takes the BQL; any thread.
 */
void orchard_block_flush(void);
void orchard_block_flush(void)
{
    bool locked = bql_locked();

    if (!locked) {
        bql_lock();
    }
    bdrv_flush_all();
    if (!locked) {
        bql_unlock();
    }
}

extern uint64_t orchard_blk_flushes, orchard_blk_errors;
extern int orchard_blk_last_errno, orchard_blk_wce;

/* See hw/block/virtio-blk.c. */
void orchard_blk_stats(uint64_t *flushes, uint64_t *errors, int *last_errno,
                       int *wce);
void orchard_blk_stats(uint64_t *flushes, uint64_t *errors, int *last_errno,
                       int *wce)
{
    *flushes = qatomic_read(&orchard_blk_flushes);
    *errors = qatomic_read(&orchard_blk_errors);
    *last_errno = qatomic_read(&orchard_blk_last_errno);
    *wce = qatomic_read(&orchard_blk_wce);
}

void orchard_display_stats(OrchardDisplayStats* out)
{
    OrchardDisplay* d = &orchard_display;

    if (out == NULL) { return; }
    memset(out, 0, sizeof(*out));
    if (!d->attached) { return; }

    QEMU_LOCK_GUARD(&d->lock);
    out->presents     = orchard_presents;
    out->refreshes    = orchard_refreshes;
    orchard_presents  = 0;
    orchard_refreshes = 0;
}

static const DisplayChangeListenerOps orchard_dcl_ops = {
    .dpy_name       = "orchard-embed",
    .dpy_refresh    = orchard_refresh,
    .dpy_gfx_update = orchard_gfx_update,
    .dpy_gfx_switch = orchard_gfx_switch,
};

void orchard_display_attach(void)
{
    OrchardDisplay* d = &orchard_display;
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
    qemu_console_register_listener(con, &orchard_dcl, &orchard_dcl_ops);

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

void orchard_display_detach(void)
{
    OrchardDisplay* d = &orchard_display;

    if (!d->attached) { return; }
    qemu_console_unregister_listener(&orchard_dcl);
    WITH_QEMU_LOCK_GUARD(&d->lock)
    {
        d->surface = NULL;
        d->dirty   = false;
    }
    d->attached = false;
}

void orchard_display_invalidate(void)
{
    OrchardDisplay* d = &orchard_display;

    if (!d->attached) { return; }
    QEMU_LOCK_GUARD(&d->lock);
    damage_all_locked(d);
}

OrchardFrameResult orchard_display_read(void* dst, size_t dst_size, OrchardFrameInfo* info)
{
    OrchardDisplay* d = &orchard_display;
    uint32_t        width, height, x0, y0, x1, y1, row;
    const uint8_t*  src;
    uint8_t*        out;
    int             src_stride;
    size_t          need, span;

    if (info == NULL) { return ORCHARD_FRAME_NONE; }
    memset(info, 0, sizeof(*info));
    if (!d->attached) { return ORCHARD_FRAME_NONE; }

    QEMU_LOCK_GUARD(&d->lock);
    if (d->surface == NULL) { return ORCHARD_FRAME_NONE; }

    width  = surface_width(d->surface);
    height = surface_height(d->surface);

    info->width      = width;
    info->height     = height;
    info->stride     = width * 4;
    info->generation = d->generation;

    need = (size_t)width * height * 4;
    /*
     * Any size but the exact one is a resize, not only a smaller one. The
     * caller sizes its buffer to the frame it was last told of, so a buffer
     * that is too big means the screen shrank: the guest going from its boot
     * framebuffer to a 960x540 desktop, say. Taken as it was, rows of the new
     * width were laid out at the old one — two copies of the picture side by
     * side, squeezed into the top of the screen.
     */
    if (dst == NULL || dst_size != need) {
        /* Report the whole screen: the caller is about to start from nothing. */
        info->w = width;
        info->h = height;
        damage_all_locked(d);
        return ORCHARD_FRAME_RESIZE;
    }
    if (!d->dirty) { return ORCHARD_FRAME_NONE; }

    x0 = MIN(d->x0, width);
    y0 = MIN(d->y0, height);
    x1 = MIN(d->x1, width);
    y1 = MIN(d->y1, height);
    if (x0 >= x1 || y0 >= y1) {
        d->dirty = false;
        return ORCHARD_FRAME_NONE;
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
    return ORCHARD_FRAME_OK;
}

/* ------------------------------------------------------------------ */
/* Input                                                               */
/* ------------------------------------------------------------------ */

void orchard_input_touch(int32_t x, int32_t y, bool pressed)
{
    OrchardDisplay* d = &orchard_display;
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

    con = orchard_dcl.con;
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

/*
 * The pointer as a trackpad drives it: an absolute position the app keeps
 * itself, a button mask (bit 0 left, bit 1 right) and wheel steps (positive
 * scrolls up). Only buttons whose state changed are sent, so holding one
 * across moves is a drag.
 */
void orchard_input_pointer(int32_t x, int32_t y, uint32_t buttons, int32_t wheel)
{
    static const InputButton map[2] = { INPUT_BUTTON_LEFT, INPUT_BUTTON_RIGHT };
    OrchardDisplay* d = &orchard_display;
    QemuConsole*    con;
    uint32_t        width = 0, height = 0, was = 0;
    int             i;

    if (!d->attached) { return; }

    WITH_QEMU_LOCK_GUARD(&d->lock)
    {
        if (d->surface == NULL) { return; }
        width  = surface_width(d->surface);
        height = surface_height(d->surface);
        was    = d->buttons;
        d->buttons = buttons & 3;
    }
    if (width == 0 || height == 0) { return; }

    x = MAX(0, MIN((int32_t)width - 1, x));
    y = MAX(0, MIN((int32_t)height - 1, y));

    con = orchard_dcl.con;
    bql_lock();
    qemu_input_queue_abs(con, INPUT_AXIS_X, x, 0, width);
    qemu_input_queue_abs(con, INPUT_AXIS_Y, y, 0, height);
    for (i = 0; i < 2; i++) {
        if (((was ^ buttons) >> i) & 1) {
            qemu_input_queue_btn(con, map[i], (buttons >> i) & 1);
        }
    }
    qemu_input_event_sync();
    for (i = 0; i < abs(wheel); i++) {
        InputButton b = wheel > 0 ? INPUT_BUTTON_WHEEL_UP : INPUT_BUTTON_WHEEL_DOWN;
        qemu_input_queue_btn(con, b, true);
        qemu_input_event_sync();
        qemu_input_queue_btn(con, b, false);
        qemu_input_event_sync();
    }
    bql_unlock();
}

void orchard_input_function_key(uint32_t number, bool pressed)
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

void orchard_input_key_hid(uint32_t usage, bool pressed)
{
    unsigned int lnx;

    if (usage >= qemu_input_map_usb_to_linux_len) { return; }
    lnx = qemu_input_map_usb_to_linux[usage];
    /* Zero is "no such key" in the table: a usage with no Linux counterpart. */
    if (lnx == 0) { return; }

    bql_lock();
    qemu_input_event_send_key_linux(orchard_dcl.con, lnx, pressed);
    bql_unlock();
}

/*
 * One whole keystroke: the modifiers in `mods` (bit 0 Shift, bit 1 Command)
 * held around a press and release of `usage`, all queued under one BQL.
 *
 * The on-screen keyboard used to send each of these as its own call, and each
 * took the BQL for itself. Under a busy guest the BQL can be held for hundreds
 * of milliseconds, so the release reached the keyboard's queue that much after
 * the press: the guest saw the key held past its repeat delay and repeated it.
 * Measured on the phone, every letter and every backspace came out twice.
 * Queued together, the release is behind the press in the same queue and the
 * guest reads it on its next poll.
 */
void orchard_input_key_tap(uint32_t usage, uint32_t mods)
{
    static const uint32_t modifier_usage[] = { 0xE1 /* LeftShift */, 0xE3 /* LeftGUI */ };
    unsigned int lnx, mod_lnx[2];
    int i;

    if (usage >= qemu_input_map_usb_to_linux_len) { return; }
    lnx = qemu_input_map_usb_to_linux[usage];
    if (lnx == 0) { return; }
    for (i = 0; i < 2; i++) {
        mod_lnx[i] = qemu_input_map_usb_to_linux[modifier_usage[i]];
    }

    bql_lock();
    for (i = 0; i < 2; i++) {
        if ((mods & (1u << i)) && mod_lnx[i]) {
            qemu_input_event_send_key_linux(orchard_dcl.con, mod_lnx[i], true);
        }
    }
    qemu_input_event_send_key_linux(orchard_dcl.con, lnx, true);
    qemu_input_event_send_key_linux(orchard_dcl.con, lnx, false);
    for (i = 1; i >= 0; i--) {
        if ((mods & (1u << i)) && mod_lnx[i]) {
            qemu_input_event_send_key_linux(orchard_dcl.con, mod_lnx[i], false);
        }
    }
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
 * ORCHARD_HEADLESS_FPS=1 attaches the same listener from inside the emulator
 * and reads frames the way the app's pump does, then prints a line a second:
 *
 *   ORCHARD-FPS <second> presents=<n> refreshes=<n>
 *
 * Off unless the variable is set, so nothing changes for the app.
 */
static void* orchard_headless_pump(void* arg)
{
    OrchardFrameInfo  info;
    OrchardDisplayStats stats;
    void*             buf  = NULL;
    size_t            size = 0;
    int64_t           next;
    int64_t           gmin = 0, gmean = 0, gmax = 0;
    uint64_t          second = 0;

    next = qemu_clock_get_ms(QEMU_CLOCK_REALTIME) + 1000;
    for (;;) {
        if (orchard_display_read(buf, size, &info) == ORCHARD_FRAME_RESIZE) {
            g_free(buf);
            size = (size_t)info.width * info.height * 4;
            buf  = g_malloc0(size);
        }
        g_usleep(1000000 / 60);
        if (qemu_clock_get_ms(QEMU_CLOCK_REALTIME) < next) { continue; }
        next += 1000;
        second++;
        orchard_display_stats(&stats);
        orchard_display_gaps(&gmin, &gmean, &gmax);
        fprintf(stderr,
                "ORCHARD-FPS %" PRIu64 " presents=%" PRIu64 " refreshes=%" PRIu64 " gap=%" PRId64 "/%" PRId64 "/%"
                PRId64 "ms\n",
                second, stats.presents, stats.refreshes, gmin, gmean, gmax);
        fflush(stderr);
    }
    return NULL;
}

static void orchard_headless_start(Notifier* n, void* opaque)
{
    static QemuThread thread;

    if (g_strcmp0(getenv("ORCHARD_HEADLESS_FPS"), "1") != 0) { return; }

    orchard_display_attach();
    qemu_thread_create(&thread, "orchard.fps", orchard_headless_pump, NULL, QEMU_THREAD_DETACHED);
}

static Notifier orchard_headless_notifier = {.notify = orchard_headless_start};

static void __attribute__((constructor)) orchard_headless_register(void)
{ qemu_add_machine_init_done_notifier(&orchard_headless_notifier); }
