/*
 * Guest display and input for an embedder that runs QEMU inside its own
 * process.
 *
 * On a phone the emulator is a library, not a program: the app calls
 * qemu_init/qemu_main_loop directly. Its display therefore has no business
 * going through a socket. The stock way to show a guest here was a VNC server
 * on the loopback with the Raw encoding, which means the framebuffer is
 * compared, encoded, written, read, decoded and blitted — six passes over five
 * megabytes, all to move pixels a few kilobytes apart in the same address
 * space.
 *
 * This is the short way round: a display change listener keeps track of what
 * the guest has redrawn, and the app copies those rows straight out of the
 * surface. Nothing is encoded, and only what changed is touched.
 *
 * Taken from Inferno-iOS (ui/inferno-embed.h), where the app shell this port
 * uses was written against it; the names carry this tree's prefix instead.
 * The app finds every one of these with dlsym. Only the declarations this
 * file defines are carried over: the original also declares iPhone hardware
 * (USB network link, battery, reset hold, taptic engine), which lives in
 * machine code this tree does not have. The app looks those up optionally
 * and does without.
 *
 * Copyright (c) 2026 Makr (Inferno-iOS, Orchard iOS port).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef UI_ORCHARD_EMBED_H
#define UI_ORCHARD_EMBED_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* What one call to orchard_display_read() found. */
typedef enum OrchardFrameResult
{
    /* The guest has not redrawn anything since the previous call. */
    ORCHARD_FRAME_NONE = 0,
    /* The damaged rows were copied; the rectangle says which. */
    ORCHARD_FRAME_OK = 1,
    /*
     * The destination is not exactly one frame, which is how a change of
     * resolution arrives — larger or smaller. The size in the info is the new
     * one; allocate exactly that and call again.
     */
    ORCHARD_FRAME_RESIZE = 2,
} OrchardFrameResult;

typedef struct OrchardFrameInfo
{
    uint32_t width;
    uint32_t height;
    /* Bytes per row of the destination, always width * 4. */
    uint32_t stride;
    /* What changed, in pixels; the whole screen after a resize. */
    uint32_t x, y, w, h;
    /* Bumped whenever the guest's screen changes size. */
    uint32_t generation;
} OrchardFrameInfo;

/*
 * Starts following the machine's graphic console. Call once qemu_init() has
 * returned and while its lock is still held — the listener list is not
 * thread-safe.
 */
void orchard_display_attach(void);
void orchard_display_detach(void);

/* Redraws everything on the next read; for when the app has lost its copy. */
void orchard_display_invalidate(void);

/*
 * Copies whatever the guest has redrawn into `dst`, which holds a whole frame
 * in a8r8g8b8 (0xAARRGGBB little-endian words). Safe to call from any thread.
 */
OrchardFrameResult orchard_display_read(void* dst, size_t dst_size, OrchardFrameInfo* info);

/*
 * Where the frames go, for when there are fewer of them than there should be.
 *
 * `presents` counts what the machine showed — every frame the display device
 * put in the console. `refreshes` counts how often QEMU's main loop got round
 * to asking the machine to redraw, which is a different thing entirely: it is
 * the one that suffers when the vCPUs are holding the big lock.
 *
 * Both are totals since the last read, and reading clears them.
 */
typedef struct OrchardDisplayStats
{
    uint64_t presents;
    uint64_t refreshes;
} OrchardDisplayStats;

void orchard_display_stats(OrchardDisplayStats* out);

/*
 * Called by the display device when it has shown a frame. Counting the
 * listener's own updates would not do: a device may report its damage a
 * row-span at a time, so one frame can arrive as a dozen of them.
 */
void orchard_display_note_present(void);

/*
 * A touch at an absolute position in framebuffer pixels. On this machine it
 * lands on the USB tablet: the pointer goes where the finger is, and the left
 * button follows the finger down and up.
 */
void orchard_input_touch(int32_t x, int32_t y, bool pressed);

/* Function keys F1..F12, by number. */
void orchard_input_function_key(uint32_t number, bool pressed);

/*
 * Any key, by its USB HID usage (page 7) — the code iOS itself reports for a
 * key, as UIKey.keyCode, so a hardware keyboard needs no table on the app's
 * side. It reaches the guest through the machine's USB keyboard. Orchard's
 * addition: the iPhone guest it was written for has no keyboard.
 */
void orchard_input_key_hid(uint32_t usage, bool pressed);
/* A whole keystroke under one BQL: `mods` bit 0 Shift, bit 1 Command. */
void orchard_input_key_tap(uint32_t usage, uint32_t mods);
/* Pointer as a trackpad drives it: position, buttons (bit 0 left, bit 1
 * right), wheel steps (positive up). */
void orchard_input_pointer(int32_t x, int32_t y, uint32_t buttons, int32_t wheel);

#ifdef __cplusplus
}
#endif

#endif /* UI_ORCHARD_EMBED_H */
