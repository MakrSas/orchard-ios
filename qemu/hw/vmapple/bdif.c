/*
 * VMApple Backdoor Interface
 *
 * Copyright © 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/units.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "trace.h"
#include "hw/vmapple/vmapple.h"
#include "hw/core/sysbus.h"
#include "hw/block/block.h"
#include "qapi/error.h"
#include "system/block-backend.h"
#include "system/dma.h"
#include "chardev/char-fe.h"
#include "hw/core/qdev-properties-system.h"

OBJECT_DECLARE_SIMPLE_TYPE(VMAppleBdifState, VMAPPLE_BDIF)

struct VMAppleBdifState {
    SysBusDevice parent_obj;

    BlockBackend *aux;
    BlockBackend *root;
    MemoryRegion mmio;
    CharFrontend usbdev;
    GByteArray *usb_input;
    uint32_t usb_queue;
    bool usb_rx_ready;
    bool usb_rx_posted;
    uint64_t usb_rx_addr;
    uint32_t usb_rx_len;
    bool usb_rx_pending;
    uint64_t usb_rx_pending_addr;
    uint32_t usb_rx_pending_len;
};

#define VMAPPLE_BDIF_SIZE   0x00200000

#define REG_DEVID_MASK      0xffff0000
#define DEVID_ROOT          0x00000000
#define DEVID_AUX           0x00010000
#define DEVID_USB           0x00100000

#define REG_STATUS          0x0
#define REG_STATUS_ACTIVE     BIT(0)
#define REG_CFG             0x4
#define REG_CFG_ACTIVE        BIT(1)
#define REG_CFG_USB           0x1a01
#define REG_UNK1            0x8
#define REG_BUSY            0x10
#define REG_BUSY_READY        BIT(0)
#define REG_UNK2            0x400
#define REG_CMD             0x408
#define REG_QUEUE           0x410
#define REG_NEXT_DEVICE     0x420
#define REG_UNK3            0x434

typedef struct VblkSector {
    uint32_t pad;
    uint32_t pad2;
    uint32_t sector;
    uint32_t pad3;
} VblkSector;

typedef struct VblkReqCmd {
    uint64_t addr;
    uint32_t len;
    uint32_t flags;
} VblkReqCmd;

typedef struct VblkReq {
    VblkReqCmd sector;
    VblkReqCmd data;
    VblkReqCmd retval;
} VblkReq;

#define VBLK_DATA_FLAGS_READ  0x00030001
#define VBLK_DATA_FLAGS_WRITE 0x00010001

#define VBLK_RET_SUCCESS  0
#define VBLK_RET_FAILED   1

static bool bdif_usb_enabled(VMAppleBdifState *s)
{
    return qemu_chr_fe_backend_connected(&s->usbdev);
}

static void vusb_reset_state(VMAppleBdifState *s)
{
    s->usb_queue = 0;
    s->usb_rx_ready = false;
    s->usb_rx_posted = false;
    s->usb_rx_addr = 0;
    s->usb_rx_len = 0;
    s->usb_rx_pending = false;
    s->usb_rx_pending_addr = 0;
    s->usb_rx_pending_len = 0;
    if (s->usb_input) {
        g_byte_array_set_size(s->usb_input, 0);
    }
}

static void vusb_try_deliver(VMAppleBdifState *s);

/*
 * Queue0 lifecycle tracer (diagnostic only, no state mutation).
 * flags bit0=posted bit1=ready bit2=pending; head_len is the leading
 * wire length word when at least 4 buffered bytes exist, else 0
 * (disambiguate via buffered byte count). Guest PC omitted: it is not
 * readily available in this MMIO/char-fe context without invasive API.
 */
static inline void bdif_usb_q0_trace(VMAppleBdifState *s, const char *event,
                                     uint64_t daddr, uint32_t dlen)
{
    uint32_t flags;
    uint32_t buffered;
    uint32_t head_len = 0;
    if (!trace_event_get_state_backends(TRACE_BDIF_USB_Q0)) {
        return;
    }
    flags = (s->usb_rx_posted ? 1u : 0u) |
            (s->usb_rx_ready ? 2u : 0u) |
            (s->usb_rx_pending ? 4u : 0u);
    buffered = (s->usb_input != NULL) ? (uint32_t)s->usb_input->len : 0;
    if (buffered >= sizeof(uint32_t) && s->usb_input != NULL) {
        uint32_t head;

        memcpy(&head, s->usb_input->data, sizeof(head));
        head_len = le32_to_cpu(head);
    }
    trace_bdif_usb_q0(event, daddr, dlen, flags, s->usb_rx_addr, s->usb_rx_len,
                      s->usb_rx_pending_addr, s->usb_rx_pending_len,
                      buffered, head_len);
}

static uint64_t bdif_read(void *opaque, hwaddr offset, unsigned size)
{
    VMAppleBdifState *s = opaque;
    uint64_t ret = -1;
    uint64_t devid = offset & REG_DEVID_MASK;

    switch (offset & ~REG_DEVID_MASK) {
    case REG_STATUS:
        ret = REG_STATUS_ACTIVE;
        break;
    case REG_CFG:
        if (devid == DEVID_USB && bdif_usb_enabled(s)) {
            ret = REG_CFG_USB;
        } else {
            ret = REG_CFG_ACTIVE;
        }
        break;
    case REG_UNK1:
        ret = 0x420;
        break;
    case REG_BUSY:
        if (devid == DEVID_USB && bdif_usb_enabled(s) && s->usb_queue == 0) {
            ret = s->usb_rx_ready ? REG_BUSY_READY : 0;
            if (s->usb_rx_ready) {
                bdif_usb_q0_trace(s, "consume", 0, 0);
                s->usb_rx_posted = false;
                s->usb_rx_ready = false;
                /*
                 * A one-shot DATA repost may have arrived before the guest
                 * consumed SETUP completion. Promote it now so buffered
                 * DATA can deliver on the next poll.
                 */
                if (s->usb_rx_pending) {
                    s->usb_rx_pending = false;
                    s->usb_rx_addr = s->usb_rx_pending_addr;
                    s->usb_rx_len = s->usb_rx_pending_len;
                    s->usb_rx_pending_addr = 0;
                    s->usb_rx_pending_len = 0;
                    s->usb_rx_posted = true;
                    vusb_try_deliver(s);
                    bdif_usb_q0_trace(s, "promote", s->usb_rx_addr,
                                      s->usb_rx_len);
                }
            }
        } else {
            ret = REG_BUSY_READY;
        }
        break;
    case REG_UNK2:
        ret = 0x1;
        break;
    case REG_QUEUE:
        if (devid == DEVID_USB && bdif_usb_enabled(s)) {
            ret = s->usb_queue;
        }
        break;
    case REG_UNK3:
        ret = 0x0;
        break;
    case REG_NEXT_DEVICE:
        switch (devid) {
        case DEVID_ROOT:
            ret = 0x8000000;
            break;
        case DEVID_AUX:
            ret = 0x10000;
            break;
        }
        break;
    }

    trace_bdif_read(offset, size, ret);
    return ret;
}

static void le2cpu_sector(VblkSector *sector)
{
    sector->sector = le32_to_cpu(sector->sector);
}

static void le2cpu_reqcmd(VblkReqCmd *cmd)
{
    cmd->addr = le64_to_cpu(cmd->addr);
    cmd->len = le32_to_cpu(cmd->len);
    cmd->flags = le32_to_cpu(cmd->flags);
}

static void le2cpu_req(VblkReq *req)
{
    le2cpu_reqcmd(&req->sector);
    le2cpu_reqcmd(&req->data);
    le2cpu_reqcmd(&req->retval);
}

static void vblk_cmd(uint64_t devid, BlockBackend *blk, uint64_t gp_addr,
                     uint64_t static_off)
{
    VblkReq req;
    VblkSector sector;
    uint64_t off = 0;
    g_autofree char *buf = NULL;
    uint8_t ret = VBLK_RET_FAILED;
    int r;
    MemTxResult dma_result;

    dma_result = dma_memory_read(&address_space_memory, gp_addr,
                                 &req, sizeof(req), MEMTXATTRS_UNSPECIFIED);
    if (dma_result != MEMTX_OK) {
        goto out;
    }

    le2cpu_req(&req);

    if (req.sector.len != sizeof(sector)) {
        goto out;
    }

    /* Read the vblk command */
    dma_result = dma_memory_read(&address_space_memory, req.sector.addr,
                                 &sector, sizeof(sector),
                                 MEMTXATTRS_UNSPECIFIED);
    if (dma_result != MEMTX_OK) {
        goto out;
    }
    le2cpu_sector(&sector);

    off = sector.sector * 512ULL + static_off;

    /* Sanity check that we're not allocating bogus sizes */
    if (req.data.len > 128 * MiB) {
        goto out;
    }

    buf = g_malloc0(req.data.len);
    switch (req.data.flags) {
    case VBLK_DATA_FLAGS_READ:
        r = blk_pread(blk, off, req.data.len, buf, 0);
        trace_bdif_vblk_read(devid == DEVID_AUX ? "aux" : "root",
                             req.data.addr, off, req.data.len, r);
        if (r < 0) {
            goto out;
        }
        dma_result = dma_memory_write(&address_space_memory, req.data.addr, buf,
                                      req.data.len, MEMTXATTRS_UNSPECIFIED);
        if (dma_result == MEMTX_OK) {
            ret = VBLK_RET_SUCCESS;
        }
        break;
    case VBLK_DATA_FLAGS_WRITE:
        /*
         * iBoot only reads, but the booted OS writes: this is the path macOS
         * uses to persist NVRAM. Dropping the write silently (as this arm used
         * to) made every variable immortal -- the guest would delete an OTA
         * marker, log the delete, and read the old value straight back from the
         * image. An interrupted install then never advances, because the
         * installer clears its phase markers as it completes each phase and
         * finds them restored on the next boot. Measured as a reboot roughly
         * every ten minutes on a mid-upgrade tart image, which the same image
         * does not do under a hypervisor that honours the write.
         */
        dma_result = dma_memory_read(&address_space_memory, req.data.addr, buf,
                                     req.data.len, MEMTXATTRS_UNSPECIFIED);
        if (dma_result != MEMTX_OK) {
            goto out;
        }
        r = blk_pwrite(blk, off, req.data.len, buf, 0);
        trace_bdif_vblk_write(devid == DEVID_AUX ? "aux" : "root",
                              req.data.addr, off, req.data.len, r);
        if (r < 0) {
            goto out;
        }
        ret = VBLK_RET_SUCCESS;
        break;
    default:
        break;
    }

out:
    dma_memory_write(&address_space_memory, req.retval.addr, &ret, 1,
                     MEMTXATTRS_UNSPECIFIED);
}

typedef struct VUsbDesc {
    uint64_t addr;
    uint32_t len;
    uint16_t next;
    uint16_t flags;
} QEMU_PACKED VUsbDesc;

#define VUSB_MAX_PACKET  (128 * KiB)
#define VUSB_MAX_TX      4096
#define VUSB_MAX_RX      (2 * (VUSB_MAX_PACKET + sizeof(uint32_t)))

static void vusb_try_deliver(VMAppleBdifState *s)
{
    uint32_t len;
    MemTxResult dma_result;

    if (!s->usb_input || !s->usb_rx_posted || s->usb_rx_ready ||
        s->usb_input->len < sizeof(len)) {
        return;
    }

    memcpy(&len, s->usb_input->data, sizeof(len));
    len = le32_to_cpu(len);
    if (len > VUSB_MAX_PACKET) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "vmapple-bdif: invalid USB packet len %u\n", len);
        bdif_usb_q0_trace(s, "drop-badlen", s->usb_rx_addr, s->usb_rx_len);
        g_byte_array_set_size(s->usb_input, 0);
        return;
    }
    if (s->usb_input->len < sizeof(len) + len) {
        return;
    }
    if (len > s->usb_rx_len) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "vmapple-bdif: USB packet %u exceeds guest buffer %u\n",
                      len, s->usb_rx_len);
        bdif_usb_q0_trace(s, "retain-oversize", s->usb_rx_addr, s->usb_rx_len);
        /* Retain oversize frame for a larger reposted buffer; the guest
         * must repost before any buffered frame is consumed. */
        return;
    }

    if (len == 0) {
        s->usb_rx_ready = true;
        bdif_usb_q0_trace(s, "deliver-zlp", s->usb_rx_addr, s->usb_rx_len);
    } else {
        dma_result = dma_memory_write(&address_space_memory, s->usb_rx_addr,
                                      s->usb_input->data + sizeof(len), len,
                                      MEMTXATTRS_UNSPECIFIED);
        if (dma_result != MEMTX_OK) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "vmapple-bdif: USB DMA write failed addr=0x%" PRIx64
                          " len=%u result=%d\n",
                          s->usb_rx_addr, len, dma_result);
            bdif_usb_q0_trace(s, "dma-fail", s->usb_rx_addr, s->usb_rx_len);
        } else {
            s->usb_rx_ready = true;
            bdif_usb_q0_trace(s, "deliver", s->usb_rx_addr, s->usb_rx_len);
        }
    }
    g_byte_array_remove_range(s->usb_input, 0, sizeof(len) + len);
}

static int vusb_can_receive(void *opaque)
{
    VMAppleBdifState *s = opaque;
    const int max_rx = VUSB_MAX_RX;

    if (!s->usb_input) {
        return 0;
    }
    if (s->usb_input->len >= (guint)max_rx) {
        return 0;
    }
    return max_rx - (int)s->usb_input->len;
}

static void vusb_receive(void *opaque, const uint8_t *buf, int size)
{
    VMAppleBdifState *s = opaque;
    const guint max_rx = VUSB_MAX_RX;

    if (!s->usb_input || size <= 0) {
        return;
    }
    if (s->usb_input->len >= max_rx) {
        return;
    }
    if (s->usb_input->len + (guint)size > max_rx) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "vmapple-bdif: USB rx overflow, dropping %d bytes\n",
                      size);
        return;
    }
    g_byte_array_append(s->usb_input, buf, size);
    bdif_usb_q0_trace(s, "rx", 0, 0);
    vusb_try_deliver(s);
}

static void vusb_event(void *opaque, QEMUChrEvent event)
{
    VMAppleBdifState *s = opaque;

    if (event == CHR_EVENT_CLOSED || event == CHR_EVENT_BREAK) {
        s->usb_rx_ready = false;
        s->usb_rx_posted = false;
        s->usb_rx_pending = false;
        s->usb_rx_pending_addr = 0;
        s->usb_rx_pending_len = 0;
        if (s->usb_input) {
            g_byte_array_set_size(s->usb_input, 0);
        }
    }
}

static void vusb_cmd(VMAppleBdifState *s, uint64_t gp_addr)
{
    uint64_t words[16];
    VUsbDesc desc;
    MemTxResult dma_result;
    unsigned i;

    dma_result = dma_memory_read(&address_space_memory, gp_addr, words,
                                 sizeof(words), MEMTXATTRS_UNSPECIFIED);
    trace_bdif_usb_cmd(gp_addr, dma_result);
    if (dma_result != MEMTX_OK) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "vmapple-bdif: USB desc fetch failed addr=0x%" PRIx64
                      " result=%d\n", gp_addr, dma_result);
        return;
    }

    for (i = 0; i < ARRAY_SIZE(words); i++) {
        trace_bdif_usb_word(i * sizeof(words[i]), le64_to_cpu(words[i]));
    }

    memcpy(&desc, words, sizeof(desc));
    desc.addr = le64_to_cpu(desc.addr);
    desc.len = le32_to_cpu(desc.len);
    desc.next = le16_to_cpu(desc.next);
    desc.flags = le16_to_cpu(desc.flags);

    if (s->usb_queue == 0) {
        if (s->usb_rx_ready) {
            if (desc.len == 0 || desc.len > VUSB_MAX_PACKET) {
                qemu_log_mask(LOG_GUEST_ERROR,
                              "vmapple-bdif: invalid USB rx buffer len %u\n",
                              desc.len);
                bdif_usb_q0_trace(s, "reject-len", desc.addr, desc.len);
                return;
            }
            /*
             * The guest may post the DATA buffer before consuming SETUP
             * completion, and CMDs are one-shot. Stash it until the BUSY
             * read consumes the ready frame.
             */
            if (s->usb_rx_pending) {
                qemu_log_mask(LOG_GUEST_ERROR,
                              "vmapple-bdif: USB rx repost overwrites pending\n");
                s->usb_rx_pending_addr = desc.addr;
                s->usb_rx_pending_len = desc.len;
                s->usb_rx_pending = true;
                bdif_usb_q0_trace(s, "stash-overwrite", desc.addr, desc.len);
            } else {
                s->usb_rx_pending_addr = desc.addr;
                s->usb_rx_pending_len = desc.len;
                s->usb_rx_pending = true;
                bdif_usb_q0_trace(s, "stash", desc.addr, desc.len);
            }
            return;
        }
        if (desc.len == 0 || desc.len > VUSB_MAX_PACKET) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "vmapple-bdif: invalid USB rx buffer len %u\n",
                          desc.len);
            bdif_usb_q0_trace(s, "reject-len", desc.addr, desc.len);
            return;
        }
        s->usb_rx_addr = desc.addr;
        s->usb_rx_len = desc.len;
        s->usb_rx_posted = true;
        bdif_usb_q0_trace(s, "post", desc.addr, desc.len);
        vusb_try_deliver(s);
    } else if (s->usb_queue == 1) {
        if (desc.len == 0 || desc.len > VUSB_MAX_TX) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "vmapple-bdif: invalid USB tx len %u\n", desc.len);
            return;
        }
        g_autofree uint8_t *buf = g_malloc(desc.len);

        dma_result = dma_memory_read(&address_space_memory, desc.addr, buf,
                                     desc.len, MEMTXATTRS_UNSPECIFIED);
        if (dma_result != MEMTX_OK) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "vmapple-bdif: USB tx DMA read failed addr=0x%"
                          PRIx64 " len=%u result=%d\n",
                          desc.addr, desc.len, dma_result);
            return;
        }
        {
            uint32_t wire_len = cpu_to_le32(desc.len);

            for (i = 0; i < DIV_ROUND_UP(desc.len, sizeof(uint64_t)); i++) {
                uint64_t word = 0;
                size_t chunk = MIN(sizeof(word), desc.len - i * sizeof(word));

                memcpy(&word, buf + i * sizeof(word), chunk);
                trace_bdif_usb_word(i * sizeof(word), le64_to_cpu(word));
            }
            if (qemu_chr_fe_backend_connected(&s->usbdev)) {
                int r;

                r = qemu_chr_fe_write_all(&s->usbdev, (uint8_t *)&wire_len,
                                          sizeof(wire_len));
                if (r != sizeof(wire_len)) {
                    qemu_log_mask(LOG_GUEST_ERROR,
                                  "vmapple-bdif: USB tx header write failed "
                                  "r=%d\n", r);
                    return;
                }
                r = qemu_chr_fe_write_all(&s->usbdev, buf, desc.len);
                if (r != (int)desc.len) {
                    qemu_log_mask(LOG_GUEST_ERROR,
                                  "vmapple-bdif: USB tx payload write failed "
                                  "r=%d len=%u\n", r, desc.len);
                }
            }
        }
    } else {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "vmapple-bdif: invalid USB queue %u\n", s->usb_queue);
    }
}

static void bdif_write(void *opaque, hwaddr offset,
                       uint64_t value, unsigned size)
{
    VMAppleBdifState *s = opaque;
    uint64_t devid = (offset & REG_DEVID_MASK);

    trace_bdif_write(offset, size, value);

    switch (offset & ~REG_DEVID_MASK) {
    case REG_QUEUE:
        if (devid == DEVID_USB && bdif_usb_enabled(s)) {
            if (value > 1) {
                qemu_log_mask(LOG_GUEST_ERROR,
                              "vmapple-bdif: invalid USB queue %"
                              PRIu64 "\n", value);
            } else {
                s->usb_queue = value;
            }
        }
        break;
    case REG_CMD:
        switch (devid) {
        case DEVID_ROOT:
            vblk_cmd(devid, s->root, value, 0x0);
            break;
        case DEVID_AUX:
            vblk_cmd(devid, s->aux, value, 0x0);
            break;
        case DEVID_USB:
            if (bdif_usb_enabled(s)) {
                vusb_cmd(s, value);
            }
            break;
        }
        break;
    }
}

static const MemoryRegionOps bdif_ops = {
    .read = bdif_read,
    .write = bdif_write,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

static void bdif_init(Object *obj)
{
    VMAppleBdifState *s = VMAPPLE_BDIF(obj);

    memory_region_init_io(&s->mmio, obj, &bdif_ops, obj,
                         "VMApple Backdoor Interface", VMAPPLE_BDIF_SIZE);
    sysbus_init_mmio(SYS_BUS_DEVICE(obj), &s->mmio);
    s->usb_input = g_byte_array_new();
}

static void bdif_reset(Object *obj, ResetType type)
{
    VMAppleBdifState *s = VMAPPLE_BDIF(obj);

    vusb_reset_state(s);
}

static void bdif_realize(DeviceState *dev, Error **errp)
{
    VMAppleBdifState *s = VMAPPLE_BDIF(dev);

    /*
     * Ask for write permission on the AUX backend.
     *
     * `DEFINE_PROP_DRIVE` attaches a backend but grants nothing, and a
     * `blk_pwrite` against an unpermitted one fails with -EPERM before it
     * reaches the image. The guest's NVRAM lives on this device, so without
     * this every variable the booted OS writes or deletes is lost and the old
     * contents read straight back -- which strands an interrupted install in a
     * loop, re-running the phase whose marker it just cleared.
     *
     * AUX only: the root backend is attached read-only by the launcher, so
     * asking for write there would fail realize for every configuration that
     * does not need it.
     */
    if (s->aux != NULL && blk_set_perm(s->aux,
                                       BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE,
                                       BLK_PERM_ALL, errp) < 0) {
        return;
    }

    qemu_chr_fe_set_handlers(&s->usbdev, vusb_can_receive, vusb_receive,
                             vusb_event, NULL, s, NULL, true);
}

static void bdif_finalize(Object *obj)
{
    VMAppleBdifState *s = VMAPPLE_BDIF(obj);

    g_clear_pointer(&s->usb_input, g_byte_array_unref);
}

static const Property bdif_properties[] = {
    DEFINE_PROP_DRIVE("aux", VMAppleBdifState, aux),
    DEFINE_PROP_DRIVE("root", VMAppleBdifState, root),
    DEFINE_PROP_CHR("usbdev", VMAppleBdifState, usbdev),
};

static void bdif_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    ResettableClass *rc = RESETTABLE_CLASS(klass);

    dc->desc = "VMApple Backdoor Interface";
    dc->realize = bdif_realize;
    device_class_set_props(dc, bdif_properties);
    rc->phases.hold = bdif_reset;
}

static const TypeInfo bdif_info = {
    .name          = TYPE_VMAPPLE_BDIF,
    .parent        = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(VMAppleBdifState),
    .instance_init = bdif_init,
    .instance_finalize = bdif_finalize,
    .class_init    = bdif_class_init,
};

static void bdif_register_types(void)
{
    type_register_static(&bdif_info);
}

type_init(bdif_register_types)
