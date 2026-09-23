/*
 * Audio output for an embedder that runs QEMU inside its own process.
 *
 * On a phone the emulator is a library, and the app owns the audio session
 * and the output unit; QEMU's own CoreAudio backend is the macOS HAL, which
 * iOS does not have. This backend is the short way round, as
 * ui/orchard-embed.c is for the display: the mixer hands it the guest's
 * sound at a fixed format, it keeps it in a ring, and the app pulls it out
 * of the ring from its render callback with orchard_audio_read().
 *
 * The format is fixed — 48 kHz, 16-bit, two channels, interleaved — and the
 * mixer resamples whatever the guest plays into it. The voice consumes at
 * real time, as the "none" backend does, whether or not anything reads: a
 * guest must not stall because the app is not listening. When the ring is
 * full the oldest audio goes, and a reader that finds it empty plays silence.
 *
 * One producer (the mixer, under the BQL) and one consumer (the app's audio
 * thread): the ring needs no lock, only ordered head and tail counters.
 *
 * Copyright (c) 2026 Makr (Orchard iOS port).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/module.h"
#include "qemu/atomic.h"
#include "qemu/audio.h"
#include "qom/object.h"

#include "audio_int.h"

#define TYPE_AUDIO_ORCHARD "audio-orchard"
OBJECT_DECLARE_SIMPLE_TYPE(AudioOrchard, AUDIO_ORCHARD)

struct AudioOrchard {
    AudioMixengBackend parent_obj;
};

typedef struct OrchardVoiceOut {
    HWVoiceOut hw;
    RateCtl rate;
} OrchardVoiceOut;

/* No microphone: a guest that records gets silence, as from "none". */
typedef struct OrchardVoiceIn {
    HWVoiceIn hw;
    RateCtl rate;
} OrchardVoiceIn;

#define ORCHARD_AUDIO_RATE      48000
#define ORCHARD_AUDIO_CHANNELS  2
/* Bytes; a power of two. About 340 ms at 48 kHz, 16-bit stereo. */
#define ORCHARD_AUDIO_RING      (64 * 1024)

static uint8_t orchard_ring[ORCHARD_AUDIO_RING];
static uint64_t orchard_head;   /* bytes ever written; producer only */
static uint64_t orchard_tail;   /* bytes ever read; consumer, or producer on overrun */
static uint64_t orchard_read_total;     /* bytes the app actually took */
static uint64_t orchard_loud_total;     /* samples written that were not silence */

/*
 * Totals for the app's log: bytes the guest's sound card sent, bytes the app
 * took, and how many of the samples sent were not zero — telling "the guest
 * plays nothing" from "the guest plays silence" from "nobody reads".
 */
void orchard_audio_stats(uint64_t *written, uint64_t *read, uint64_t *loud);
void orchard_audio_stats(uint64_t *written, uint64_t *read, uint64_t *loud)
{
    *written = qatomic_read(&orchard_head);
    *read = qatomic_read(&orchard_read_total);
    *loud = qatomic_read(&orchard_loud_total);
}

/* The format orchard_audio_read() returns: rate and channel count. */
void orchard_audio_format(uint32_t *rate, uint32_t *channels);
void orchard_audio_format(uint32_t *rate, uint32_t *channels)
{
    *rate = ORCHARD_AUDIO_RATE;
    *channels = ORCHARD_AUDIO_CHANNELS;
}

/* Bytes waiting in the ring, for the reader's jitter buffer. */
size_t orchard_audio_available(void);
size_t orchard_audio_available(void)
{
    return qatomic_load_acquire(&orchard_head) - qatomic_read(&orchard_tail);
}

/*
 * Up to @bytes of interleaved 16-bit samples into @dst; returns how many were
 * there. Called from the app's real-time audio thread: no locks, no waits.
 */
size_t orchard_audio_read(void *dst, size_t bytes);
size_t orchard_audio_read(void *dst, size_t bytes)
{
    uint64_t head = qatomic_load_acquire(&orchard_head);
    uint64_t tail = qatomic_read(&orchard_tail);
    size_t have = head - tail;
    size_t n = MIN(have, bytes) & ~(size_t)3;   /* whole frames */
    size_t off = tail & (ORCHARD_AUDIO_RING - 1);
    size_t first = MIN(n, (size_t)ORCHARD_AUDIO_RING - off);

    memcpy(dst, orchard_ring + off, first);
    memcpy((uint8_t *)dst + first, orchard_ring, n - first);
    /* Only moves forward: an overrun on the other side may have moved it further. */
    qatomic_cmpxchg(&orchard_tail, tail, tail + n);
    qatomic_set(&orchard_read_total, qatomic_read(&orchard_read_total) + n);
    return n;
}

static void orchard_ring_put(const uint8_t *src, size_t len)
{
    uint64_t head = orchard_head;
    const int16_t *samples = (const int16_t *)src;
    uint64_t loud = 0;

    for (size_t i = 0; i < len / 2; i++) {
        loud += samples[i] != 0;
    }
    qatomic_set(&orchard_loud_total, orchard_loud_total + loud);

    if (len > ORCHARD_AUDIO_RING) {
        src += len - ORCHARD_AUDIO_RING;
        len = ORCHARD_AUDIO_RING;
    }
    for (size_t done = 0; done < len;) {
        size_t off = (head + done) & (ORCHARD_AUDIO_RING - 1);
        size_t chunk = MIN(len - done, (size_t)ORCHARD_AUDIO_RING - off);

        memcpy(orchard_ring + off, src + done, chunk);
        done += chunk;
    }
    qatomic_store_release(&orchard_head, head + len);

    /* Full: drop the oldest audio rather than hold the guest up. */
    for (;;) {
        uint64_t tail = qatomic_read(&orchard_tail);

        if (head + len - tail <= ORCHARD_AUDIO_RING) {
            break;
        }
        if (qatomic_cmpxchg(&orchard_tail, tail, head + len - ORCHARD_AUDIO_RING) == tail) {
            break;
        }
    }
}

static size_t orchard_write(HWVoiceOut *hw, void *buf, size_t len)
{
    OrchardVoiceOut *v = (OrchardVoiceOut *)hw;
    size_t bytes = audio_rate_get_bytes(&v->rate, &hw->info, len);

    orchard_ring_put(buf, bytes);
    return bytes;
}

static int orchard_init_out(HWVoiceOut *hw, struct audsettings *as)
{
    OrchardVoiceOut *v = (OrchardVoiceOut *)hw;
    struct audsettings fixed = {
        .freq = ORCHARD_AUDIO_RATE,
        .nchannels = ORCHARD_AUDIO_CHANNELS,
        .fmt = AUDIO_FORMAT_S16,
        .big_endian = false,
    };

    audio_pcm_init_info(&hw->info, &fixed);
    hw->samples = 1024;
    audio_rate_start(&v->rate);
    return 0;
}

static void orchard_fini_out(HWVoiceOut *hw)
{
}

static void orchard_enable_out(HWVoiceOut *hw, bool enable)
{
    OrchardVoiceOut *v = (OrchardVoiceOut *)hw;

    if (enable) {
        audio_rate_start(&v->rate);
    }
}

static int orchard_init_in(HWVoiceIn *hw, struct audsettings *as)
{
    OrchardVoiceIn *v = (OrchardVoiceIn *)hw;

    audio_pcm_init_info(&hw->info, as);
    hw->samples = 1024;
    audio_rate_start(&v->rate);
    return 0;
}

static void orchard_fini_in(HWVoiceIn *hw)
{
}

static size_t orchard_read(HWVoiceIn *hw, void *buf, size_t size)
{
    OrchardVoiceIn *v = (OrchardVoiceIn *)hw;
    int64_t bytes = audio_rate_get_bytes(&v->rate, &hw->info, size);

    audio_pcm_info_clear_buf(&hw->info, buf, bytes / hw->info.bytes_per_frame);
    return bytes;
}

static void orchard_enable_in(HWVoiceIn *hw, bool enable)
{
    OrchardVoiceIn *v = (OrchardVoiceIn *)hw;

    if (enable) {
        audio_rate_start(&v->rate);
    }
}

static void audio_orchard_class_init(ObjectClass *klass, const void *data)
{
    AudioMixengBackendClass *k = AUDIO_MIXENG_BACKEND_CLASS(klass);

    k->max_voices_out = 1;
    k->max_voices_in = INT_MAX;
    k->voice_size_out = sizeof(OrchardVoiceOut);
    k->voice_size_in = sizeof(OrchardVoiceIn);

    k->init_out = orchard_init_out;
    k->fini_out = orchard_fini_out;
    k->write = orchard_write;
    k->buffer_get_free = audio_generic_buffer_get_free;
    k->run_buffer_out = audio_generic_run_buffer_out;
    k->enable_out = orchard_enable_out;

    k->init_in = orchard_init_in;
    k->fini_in = orchard_fini_in;
    k->read = orchard_read;
    k->run_buffer_in = audio_generic_run_buffer_in;
    k->enable_in = orchard_enable_in;
}

static const TypeInfo audio_types[] = {
    {
        .name = TYPE_AUDIO_ORCHARD,
        .parent = TYPE_AUDIO_MIXENG_BACKEND,
        .instance_size = sizeof(AudioOrchard),
        .class_init = audio_orchard_class_init,
    },
};

DEFINE_TYPES(audio_types)
module_obj(TYPE_AUDIO_ORCHARD);
