/*
 * ASPEED USB Device Controller (UDC)
 *
 * Copyright (c) 2026 ASPEED Technology Inc.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Models the ASPEED USB Device Controller (UDC). It implements one control
 * endpoint (EP0) and 4 programmable endpoints.
 *
 * The model has two faces:
 *   - a SysBus device exposing the MMIO register interface, the interrupt and
 *     the integrated DMA engine to the guest gadget driver;
 *   - a USBDevice presented on a host controller's bus, which forwards host
 *     transactions to the guest gadget driver by raising the matching
 *     controller interrupts and completes them once the guest gadget driver
 *     responds via MMIO.
 */

#include "qemu/osdep.h"
#include "qemu/error-report.h"
#include "qemu/log.h"
#include "hw/core/irq.h"
#include "hw/core/registerfields.h"
#include "hw/core/qdev-properties.h"
#include "hw/usb/aspeed-udc.h"
#include "qemu/module.h"
#include "qapi/error.h"
#include "system/dma.h"
#include "system/address-spaces.h"
#include "trace.h"

/* Root / Global registers (offset from the controller base) */
REG32(UDC_FUNC_CTRL, 0x00)
    FIELD(UDC_FUNC_CTRL, UPSTREAM_EN,       0, 1)
REG32(UDC_IER, 0x08)
REG32(UDC_ISR, 0x0C)
    FIELD(UDC_ISR, EP_POOL_ACK,            16, 1)
    FIELD(UDC_ISR, BUS_RESET,               6, 1)
    FIELD(UDC_ISR, EP0_IN_ACK,              3, 1)
    FIELD(UDC_ISR, EP0_OUT_ACK,             1, 1)
    FIELD(UDC_ISR, EP0_SETUP,               0, 1)
REG32(UDC_EP_ACK_IER, 0x10)
REG32(UDC_EP_NAK_IER, 0x14)
REG32(UDC_EP_ACK_ISR, 0x18)
REG32(UDC_EP_NAK_ISR, 0x1C)
REG32(UDC_DEV_RESET, 0x20)
    FIELD(UDC_DEV_RESET, EP_POOL,           9, 1)
    FIELD(UDC_DEV_RESET, DMA,               8, 1)
    FIELD(UDC_DEV_RESET, ROOT,              0, 1)
REG32(UDC_STS, 0x24)
    FIELD(UDC_STS, HIGHSPEED,              27, 1)
REG32(UDC_EP0_CTRL, 0x30)
    FIELD(UDC_EP0_CTRL, RX_LEN,            16, 7)
    FIELD(UDC_EP0_CTRL, TX_LEN,             8, 7)
    FIELD(UDC_EP0_CTRL, RX_RDY,             2, 1)
    FIELD(UDC_EP0_CTRL, TX_RDY,             1, 1)
    FIELD(UDC_EP0_CTRL, STALL,              0, 1)
REG32(UDC_EP0_DATA_BUFF, 0x34)
    FIELD(UDC_EP0_DATA_BUFF, BASE_ADDR,     0, 31)
/* EP0 SETUP packet buffer: SETUP0 = bytes 0...3, SETUP1 = bytes 4...7 */
REG32(UDC_SETUP0, 0x80)
REG32(UDC_SETUP1, 0x84)

/* Per programmable-endpoint registers (offset from the EP register base) */
REG32(EP_CONFIG, 0x00)
    FIELD(EP_CONFIG, MAX_PKT,              16, 10)
    FIELD(EP_CONFIG, EP_NUM,                8, 4)
    FIELD(EP_CONFIG, DIR_OUT,               4, 1)
    FIELD(EP_CONFIG, ENABLE,                0, 1)
REG32(EP_DMA_CTRL, 0x04)
    FIELD(EP_DMA_CTRL, PROC_STS,            4, 4)
    FIELD(EP_DMA_CTRL, DESC_OP_EN,          0, 1)
REG32(EP_DMA_BUFF, 0x08)
    FIELD(EP_DMA_BUFF, BASE_ADDR,           0, 31)
REG32(EP_DMA_STS, 0x0C)
    FIELD(EP_DMA_STS, PKT_SIZE,            16, 11)
    FIELD(EP_DMA_STS, RPTR,                 8, 8)
    FIELD(EP_DMA_STS, WPTR,                 0, 8)

#define ASPEED_UDC_EP0_MAXPKT   64
#define ASPEED_UDC_EP_MAXPKT    1024

/* DMA descriptor ring (256-stage mode) and descriptor data limits */
#define ASPEED_UDC_DESCS_COUNT  256
#define ASPEED_UDC_DESC_MAX_LEN 4096

/* DMA processing-status idle codes */
#define EP_DMA_CTRL_STS_RX_IDLE 0x0
#define EP_DMA_CTRL_STS_TX_IDLE 0x8

/* DMA descriptor (DES1) fields, in guest memory */
#define ASPEED_EP_DESC1_IN_LEN(ctrl)   ((ctrl) & 0x1fff)
/* interrupt-on-completion */
#define ASPEED_EP_DESC1_INTR           BIT(31)

/* Result of moving a host data packet through an endpoint's DMA */
typedef enum {
    /* whole packet transferred */
    ASPEED_UDC_XFER_DONE,
    /* not finished, keep parked */
    ASPEED_UDC_XFER_MORE,
    /* DMA failed */
    ASPEED_UDC_XFER_ERROR,
} AspeedUDCXferResult;

static void aspeed_udc_update_irq(AspeedUDCState *s)
{
    bool level;

    level = (s->regs[R_UDC_ISR] & s->regs[R_UDC_IER]) ||
            (s->regs[R_UDC_EP_ACK_ISR] & s->regs[R_UDC_EP_ACK_IER]) ||
            (s->regs[R_UDC_EP_NAK_ISR] & s->regs[R_UDC_EP_NAK_IER]);

    trace_aspeed_udc_irq(s->regs[R_UDC_ISR], s->regs[R_UDC_IER], level);
    qemu_set_irq(s->irq, level);
}

static void aspeed_udc_raise_isr(AspeedUDCState *s, uint32_t mask)
{
    s->regs[R_UDC_ISR] |= mask;
    aspeed_udc_update_irq(s);
}

static void aspeed_udc_raise_ep_ack(AspeedUDCState *s, int ep)
{
    trace_aspeed_udc_ep_ack(ep);
    s->regs[R_UDC_EP_ACK_ISR] |= BIT(ep);
    s->regs[R_UDC_ISR] |= R_UDC_ISR_EP_POOL_ACK_MASK;
    aspeed_udc_update_irq(s);
}

/*
 * System bus device: MMIO register interface (guest gadget-driver facing)
 */

/* Connect/disconnect the gadget device from the host bus */
static void aspeed_udc_set_pullup(AspeedUDCState *s, bool on)
{
    USBDevice *udev;
    Error *err = NULL;

    if (!s->usbgadget) {
        /* no gadget device bound to this controller */
        return;
    }

    udev = USB_DEVICE(s->usbgadget);
    if (!udev->port) {
        /* not attached to a host controller bus */
        return;
    }

    trace_aspeed_udc_pullup(on, udev->attached);
    if (on && !udev->attached) {
        usb_device_attach(udev, &err);
        if (err) {
            warn_report_err(err);
        }
    } else if (!on && udev->attached) {
        usb_device_detach(udev);
    }
}

/* Complete the in-flight EP0 control transfer back to the host */
static void aspeed_udc_ep0_complete(AspeedUDCState *s, uint32_t len)
{
    USBPacket *p = s->ep0_packet;

    if (!p) {
        return;
    }

    s->ep0_packet = NULL;
    p->actual_length = s->ep0_dir_in ? MIN(len, s->ep0_setup_len)
                                     : s->ep0_setup_len;
    p->status = USB_RET_SUCCESS;
    trace_aspeed_udc_ep0_complete(s->ep0_dir_in, p->actual_length);
    usb_generic_async_ctrl_complete(USB_DEVICE(s->usbgadget), p);
}

static void aspeed_udc_ep0_tx_ready(AspeedUDCState *s, uint32_t val)
{
    uint32_t txlen = FIELD_EX32(val, UDC_EP0_CTRL, TX_LEN);
    uint32_t data_buf_addr = s->regs[R_UDC_EP0_DATA_BUFF];
    USBPacket *p;
    uint32_t n;

    if (!s->ep0_dir_in) {
        /* Status stage IN (zero length) for an OUT / no-data transfer */
        aspeed_udc_raise_isr(s, R_UDC_ISR_EP0_IN_ACK_MASK);
        aspeed_udc_ep0_complete(s, s->ep0_offset);
        return;
    }
    /* no control transfer is waiting: nothing to send */
    if (!s->ep0_packet) {
        return;
    }

    /* IN data stage: copy from the guest gadget driver's DMA buffer */
    n = MIN(txlen, s->ep0_setup_len - s->ep0_offset);
    if (n && address_space_read(&s->dram_as, data_buf_addr,
                                MEMTXATTRS_UNSPECIFIED,
                                s->ep0_data + s->ep0_offset,
                                n) != MEMTX_OK) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: EP0 IN DMA read failed\n", __func__);
        p = s->ep0_packet;
        s->ep0_packet = NULL;
        p->status = USB_RET_IOERROR;
        usb_generic_async_ctrl_complete(USB_DEVICE(s->usbgadget), p);
        return;
    }
    s->ep0_offset += n;
    aspeed_udc_raise_isr(s, R_UDC_ISR_EP0_IN_ACK_MASK);
    if (txlen < ASPEED_UDC_EP0_MAXPKT || s->ep0_offset >= s->ep0_setup_len) {
        aspeed_udc_ep0_complete(s, s->ep0_offset);
    }
}

static void aspeed_udc_ep0_rx_ready(AspeedUDCState *s)
{
    uint32_t data_buf_addr = s->regs[R_UDC_EP0_DATA_BUFF];
    USBPacket *p;
    uint32_t n;

    if (s->ep0_dir_in) {
        /* Status stage OUT (zero length) for an IN transfer */
        aspeed_udc_raise_isr(s, R_UDC_ISR_EP0_OUT_ACK_MASK);
        return;
    }
    /* no control transfer is waiting: nothing to receive */
    if (!s->ep0_packet) {
        return;
    }

    /* OUT data stage: hand host data to the guest gadget driver */
    n = MIN(s->ep0_setup_len - s->ep0_offset, ASPEED_UDC_EP0_MAXPKT);
    if (n && address_space_write(&s->dram_as, data_buf_addr,
                                 MEMTXATTRS_UNSPECIFIED,
                                 s->ep0_data + s->ep0_offset,
                                 n) != MEMTX_OK) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: EP0 OUT DMA write failed\n", __func__);
        p = s->ep0_packet;
        s->ep0_packet = NULL;
        p->status = USB_RET_IOERROR;
        usb_generic_async_ctrl_complete(USB_DEVICE(s->usbgadget), p);
        return;
    }
    s->ep0_offset += n;
    s->regs[R_UDC_EP0_CTRL] = FIELD_DP32(s->regs[R_UDC_EP0_CTRL],
                                         UDC_EP0_CTRL, RX_LEN, n);
    aspeed_udc_raise_isr(s, R_UDC_ISR_EP0_OUT_ACK_MASK);
}

/*
 * The guest gadget driver drives EP0 by writing UDC_EP0_CTRL. Translate
 * those writes into data movement to/from the deferred host control packet
 * plus the matching ACK interrupts the guest gadget driver expects.
 */
static void aspeed_udc_ep0_ctrl_write(AspeedUDCState *s, uint32_t val)
{
    USBPacket *p;

    trace_aspeed_udc_ep0_ctrl_write(val, s->ep0_dir_in, s->ep0_offset);

    if (val & R_UDC_EP0_CTRL_STALL_MASK) {
        /* Gadget stalled EP0: fail the pending control transfer */
        if (s->ep0_packet) {
            p = s->ep0_packet;
            s->ep0_packet = NULL;
            p->status = USB_RET_STALL;
            usb_generic_async_ctrl_complete(USB_DEVICE(s->usbgadget), p);
        }
    } else if (val & R_UDC_EP0_CTRL_TX_RDY_MASK) {
        s->regs[R_UDC_EP0_CTRL] &= ~R_UDC_EP0_CTRL_TX_RDY_MASK;
        aspeed_udc_ep0_tx_ready(s, val);
    } else if (val & R_UDC_EP0_CTRL_RX_RDY_MASK) {
        s->regs[R_UDC_EP0_CTRL] &= ~R_UDC_EP0_CTRL_RX_RDY_MASK;
        aspeed_udc_ep0_rx_ready(s);
    }
}

/* The upstream-enable bit connects/disconnects the gadget device */
static void aspeed_udc_func_ctrl_write(AspeedUDCState *s, uint32_t val)
{
    bool was_on = FIELD_EX32(s->regs[R_UDC_FUNC_CTRL],
                             UDC_FUNC_CTRL, UPSTREAM_EN);
    bool now_on = FIELD_EX32(val, UDC_FUNC_CTRL, UPSTREAM_EN);

    if (now_on != was_on) {
        aspeed_udc_set_pullup(s, now_on);
    }
}

static uint64_t aspeed_udc_read(void *opaque, hwaddr offset, unsigned size)
{
    AspeedUDCState *s = ASPEED_UDC(opaque);
    uint32_t reg = offset >> 2;
    uint32_t val;

    val = s->regs[reg];
    trace_aspeed_udc_read(offset, val);

    return val;
}

static void aspeed_udc_write(void *opaque, hwaddr offset, uint64_t data,
                             unsigned size)
{
    AspeedUDCState *s = ASPEED_UDC(opaque);
    uint32_t reg = offset >> 2;
    uint32_t val = data;

    trace_aspeed_udc_write(offset, val);

    switch (reg) {
    case R_UDC_FUNC_CTRL:
        val &= 0x000e1fff;
        aspeed_udc_func_ctrl_write(s, val);
        s->regs[R_UDC_FUNC_CTRL] = val;
        break;
    case R_UDC_IER:
    case R_UDC_EP_ACK_IER:
    case R_UDC_EP_NAK_IER:
        s->regs[reg] = val;
        aspeed_udc_update_irq(s);
        break;
    case R_UDC_ISR:
    case R_UDC_EP_ACK_ISR:
    case R_UDC_EP_NAK_ISR:
        s->regs[reg] &= ~val;
        aspeed_udc_update_irq(s);
        break;
    case R_UDC_EP0_CTRL:
        s->regs[reg] = val & (R_UDC_EP0_CTRL_STALL_MASK |
                              R_UDC_EP0_CTRL_TX_RDY_MASK |
                              R_UDC_EP0_CTRL_RX_RDY_MASK |
                              R_UDC_EP0_CTRL_TX_LEN_MASK);
        aspeed_udc_ep0_ctrl_write(s, val);
        break;
    case R_UDC_EP0_DATA_BUFF:
        s->regs[reg] = val & R_UDC_EP0_DATA_BUFF_BASE_ADDR_MASK;
        break;
    default:
        s->regs[reg] = val;
        break;
    }
}

static const MemoryRegionOps aspeed_udc_ops = {
    .read = aspeed_udc_read,
    .write = aspeed_udc_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 4,
    },
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/*
 * Copy len bytes from guest memory at addr into the IN packet, going through
 * a bounce buffer one buf-full at a time. Returns false on DMA failure.
 */
static bool aspeed_udc_ep_copy_to_pkt(AspeedUDCState *s, int ep, uint32_t addr,
                                      uint32_t len, USBPacket *p)
{
    uint8_t buf[ASPEED_UDC_EP_MAXPKT];
    uint32_t copied = 0;
    uint32_t seg;

    while (copied < len) {
        seg = MIN(len - copied, sizeof(buf));
        if (address_space_read(&s->dram_as, addr + copied,
                               MEMTXATTRS_UNSPECIFIED, buf, seg) != MEMTX_OK) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "%s: ep %d IN data DMA read failed\n", __func__,
                          ep);
            return false;
        }
        usb_packet_copy(p, buf, seg);
        copied += seg;
    }

    return true;
}

/*
 * IN transfer: send data to the host by filling its IN packet from the
 * buffers the guest gadget driver queued in the descriptor ring (from the
 * read pointer to the write pointer).
 *
 * One host packet can be bigger than one descriptor's buffer, so we copy from
 * several descriptors in a row until the packet is full or the ring is empty.
 * If a descriptor is too big for the space left in the packet, we copy only
 * part of it now and copy the rest on the next call; desc_off remembers how
 * far we got. We move the read pointer to the next descriptor only after a
 * descriptor is fully copied, so the guest gadget driver can read the pointer
 * and see how much was sent.
 *
 * This function raises the endpoint ACK by itself when the ring becomes empty
 * or when a descriptor asks for an interrupt.
 */
static AspeedUDCXferResult aspeed_udc_ep_xfer_in(AspeedUDCState *s, int ep,
                                                 USBPacket *p)
{
    QEMUIOVector *pktiov = p->combined ? &p->combined->iov : &p->iov;
    AspeedUDCEP *e = &s->ep[ep];
    uint32_t mps = FIELD_EX32(e->regs[R_EP_CONFIG], EP_CONFIG, MAX_PKT);
    uint32_t wptr = FIELD_EX32(e->regs[R_EP_DMA_STS], EP_DMA_STS, WPTR);
    uint32_t rptr = FIELD_EX32(e->regs[R_EP_DMA_STS], EP_DMA_STS, RPTR);
    uint32_t desc_base = e->regs[R_EP_DMA_BUFF];
    uint32_t desc_addr;
    uint32_t remaining;
    uint32_t desc_ctrl;
    uint32_t pkt_space;
    /* des_0: data buffer base address, des_1: control/status */
    uint32_t desc[2];
    uint32_t offset;
    uint32_t chunk;
    uint32_t dlen;
    bool done = false;
    bool ack = false;

    if (mps == 0) {
        /* a MAX_PKT field of 0 means the maximum packet size */
        mps = ASPEED_UDC_EP_MAXPKT;
    }

    trace_aspeed_udc_ep_data_in(ep, rptr, wptr, pktiov->size);

    /* walk the queued descriptors, filling the packet */
    while (rptr != wptr) {
        if (address_space_read(&s->dram_as, desc_base + rptr * sizeof(desc),
                               MEMTXATTRS_UNSPECIFIED, desc,
                               sizeof(desc)) != MEMTX_OK) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "%s: ep %d descriptor DMA read failed\n",
                          __func__, ep);
            return ASPEED_UDC_XFER_ERROR;
        }
        desc_addr = le32_to_cpu(desc[0]) & R_EP_DMA_BUFF_BASE_ADDR_MASK;
        desc_ctrl = le32_to_cpu(desc[1]);
        dlen = ASPEED_EP_DESC1_IN_LEN(desc_ctrl);
        offset = e->desc_off;
        /* how much to copy: min(descriptor bytes left, packet space left) */
        remaining = dlen > offset ? dlen - offset : 0;
        pkt_space = pktiov->size > (uint32_t)p->actual_length ?
                    pktiov->size - (uint32_t)p->actual_length : 0;
        chunk = MIN(remaining, pkt_space);

        if (!aspeed_udc_ep_copy_to_pkt(s, ep, desc_addr + offset, chunk, p)) {
            return ASPEED_UDC_XFER_ERROR;
        }
        e->desc_off += chunk;

        if (e->desc_off < dlen) {
            /*
             * The packet ran out of space in the middle of this descriptor,
             * so only part of it was copied. Stop here, and leave the read
             * pointer on this descriptor: the next call resumes copying the
             * rest (desc_off remembers how far we got).
             */
            done = true;
            break;
        }

        /*
         * This descriptor was copied in full. Advance the read pointer to the
         * next descriptor and reset desc_off so it starts from the beginning.
         */
        rptr = (rptr + 1) % ASPEED_UDC_DESCS_COUNT;
        e->desc_off = 0;
        if (desc_ctrl & ASPEED_EP_DESC1_INTR) {
            ack = true;
        }
        /*
         * This descriptor is shorter than the max packet size, i.e. a short
         * (or zero-length) packet. In USB that marks the end of the transfer,
         * so stop here.
         */
        if (dlen < mps) {
            done = true;
            break;
        }
        /*
         * The packet is now completely full, so the host has received all the
         * data it asked for. Stop here.
         */
        if ((uint32_t)p->actual_length >= pktiov->size) {
            done = true;
            break;
        }
    }

    e->regs[R_EP_DMA_STS] = FIELD_DP32(e->regs[R_EP_DMA_STS], EP_DMA_STS,
                                       RPTR, rptr);
    e->regs[R_EP_DMA_CTRL] = FIELD_DP32(e->regs[R_EP_DMA_CTRL], EP_DMA_CTRL,
                                        PROC_STS, EP_DMA_CTRL_STS_TX_IDLE);
    /* The guest gadget driver completes its request when the ring drains */
    if (rptr == wptr) {
        ack = true;
    }
    if (ack) {
        aspeed_udc_raise_ep_ack(s, ep);
    }

    return done ? ASPEED_UDC_XFER_DONE : ASPEED_UDC_XFER_MORE;
}

/*
 * OUT transfer: receive data from the host by copying its OUT packet into the
 * buffer the guest gadget driver set up (single-stage mode).
 *
 * A host packet can be bigger than one buffer, so we copy at most PKT_SIZE
 * bytes per call, continuing from where the last call stopped
 * (p->actual_length). The caller keeps the packet parked until it is fully
 * copied.
 */
static AspeedUDCXferResult aspeed_udc_ep_xfer_out(AspeedUDCState *s, int ep,
                                                  USBPacket *p)
{
    AspeedUDCEP *e = &s->ep[ep];
    uint32_t chunk = FIELD_EX32(e->regs[R_EP_DMA_STS], EP_DMA_STS, PKT_SIZE);
    uint32_t remaining = p->iov.size - (uint32_t)p->actual_length;
    uint32_t data_buf_addr = e->regs[R_EP_DMA_BUFF];
    uint32_t len = MIN(remaining, chunk);
    g_autofree uint8_t *buf = g_malloc(len);

    if (data_buf_addr && len) {
        usb_packet_copy(p, buf, len);
        if (address_space_write(&s->dram_as, data_buf_addr,
                                MEMTXATTRS_UNSPECIFIED, buf,
                                len) != MEMTX_OK) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "%s: ep %d OUT data DMA write failed\n",
                          __func__, ep);
            return ASPEED_UDC_XFER_ERROR;
        }
    }

    e->regs[R_EP_DMA_STS] = FIELD_DP32(e->regs[R_EP_DMA_STS],
                                       EP_DMA_STS, PKT_SIZE, len);
    e->regs[R_EP_DMA_STS] = FIELD_DP32(e->regs[R_EP_DMA_STS],
                                       EP_DMA_STS, WPTR, 0);
    e->regs[R_EP_DMA_CTRL] = FIELD_DP32(e->regs[R_EP_DMA_CTRL], EP_DMA_CTRL,
                                        PROC_STS, EP_DMA_CTRL_STS_RX_IDLE);
    aspeed_udc_raise_ep_ack(s, ep);

    if ((uint32_t)p->actual_length >= p->iov.size) {
        return ASPEED_UDC_XFER_DONE;
    }

    return ASPEED_UDC_XFER_MORE;
}

/*
 * IN kick: the guest gadget driver wrote EP_DMA_STS to tell us it queued more
 * IN data to send to the host. If a host IN request is already waiting
 * (parked because there was no data before), send the data now and finish it.
 * If the request needs more data than was queued, keep it parked and wait for
 * the next kick.
 */
static void aspeed_udc_ep_in_kick(AspeedUDCState *s, int ep, uint32_t val)
{
    AspeedUDCEP *e = &s->ep[ep];
    uint32_t cur_rptr = FIELD_EX32(e->regs[R_EP_DMA_STS], EP_DMA_STS, RPTR);
    uint32_t new_rptr = FIELD_EX32(val, EP_DMA_STS, RPTR);
    uint32_t new_wptr = FIELD_EX32(val, EP_DMA_STS, WPTR);
    USBPacket *p = e->pkt;

    /*
     * A normal kick only sets the write pointer and leaves the read-pointer
     * field 0 (the read pointer is ours to advance). The guest resets the ring
     * by writing a read pointer that is non-zero and equal to the write
     * pointer.
     *
     * We check non-zero as well as equal: on a normal kick whose write pointer
     * just wrapped back to 0, both fields would be 0, so an "equal" test alone
     * would look like a reset by mistake.
     */
    if (new_rptr != 0 && new_rptr == new_wptr) {
        cur_rptr = new_rptr;
        e->desc_off = 0;
    }
    /* store the guest's write, but keep our own read pointer */
    e->regs[R_EP_DMA_STS] = FIELD_DP32(val, EP_DMA_STS, RPTR, cur_rptr);

    /* nothing to do unless an IN packet is waiting and the ring has data */
    if (!p || cur_rptr == new_wptr) {
        return;
    }

    switch (aspeed_udc_ep_xfer_in(s, ep, p)) {
    case ASPEED_UDC_XFER_DONE:
        e->pkt = NULL;
        p->status = USB_RET_SUCCESS;
        usb_packet_complete(USB_DEVICE(s->usbgadget), p);
        break;
    case ASPEED_UDC_XFER_ERROR:
        e->pkt = NULL;
        p->status = USB_RET_IOERROR;
        usb_packet_complete(USB_DEVICE(s->usbgadget), p);
        break;
    case ASPEED_UDC_XFER_MORE:
        break;
    }
}

/*
 * OUT kick: the guest gadget driver wrote EP_DMA_STS to give us a buffer for
 * OUT data. If an OUT packet is already waiting (parked because there was no
 * buffer before), copy its data into the buffer now and finish it. If the
 * packet has more data than fits, keep it parked and wait for the next buffer.
 */
static void aspeed_udc_ep_out_kick(AspeedUDCState *s, int ep)
{
    AspeedUDCEP *e = &s->ep[ep];
    USBPacket *p = e->pkt;

    /* nothing to do unless an OUT packet is waiting and a buffer is ready */
    if (!p || !FIELD_EX32(e->regs[R_EP_DMA_STS], EP_DMA_STS, WPTR)) {
        return;
    }

    switch (aspeed_udc_ep_xfer_out(s, ep, p)) {
    case ASPEED_UDC_XFER_DONE:
        e->pkt = NULL;
        p->status = USB_RET_SUCCESS;
        usb_packet_complete(USB_DEVICE(s->usbgadget), p);
        break;
    case ASPEED_UDC_XFER_ERROR:
        e->pkt = NULL;
        p->status = USB_RET_IOERROR;
        usb_packet_complete(USB_DEVICE(s->usbgadget), p);
        break;
    case ASPEED_UDC_XFER_MORE:
        break;
    }
}

static uint64_t aspeed_udc_ep_read(void *opaque, hwaddr offset, unsigned size)
{
    AspeedUDCEP *e = opaque;
    uint32_t reg = offset >> 2;
    uint32_t val;

    val = e->regs[reg];
    trace_aspeed_udc_ep_read(e->index, offset, val);

    return val;
}

static void aspeed_udc_ep_write(void *opaque, hwaddr offset, uint64_t data,
                                unsigned size)
{
    AspeedUDCEP *e = opaque;
    AspeedUDCState *s = container_of(e - e->index, AspeedUDCState, ep[0]);
    uint32_t reg = offset >> 2;
    uint32_t val = data;

    trace_aspeed_udc_ep_write(e->index, offset, val);

    switch (reg) {
    case R_EP_DMA_BUFF:
        e->regs[reg] = val & R_EP_DMA_BUFF_BASE_ADDR_MASK;
        break;
    case R_EP_DMA_STS:
        val &= 0x77ffffff;
        if (FIELD_EX32(e->regs[R_EP_DMA_CTRL], EP_DMA_CTRL, DESC_OP_EN)) {
            /* IN, descriptor-list mode */
            aspeed_udc_ep_in_kick(s, e->index, val);
        } else {
            /* OUT, single-stage mode */
            e->regs[reg] = val;
            aspeed_udc_ep_out_kick(s, e->index);
        }
        break;
    default:
        e->regs[reg] = val;
        break;
    }
}

static const MemoryRegionOps aspeed_udc_ep_ops = {
    .read = aspeed_udc_ep_read,
    .write = aspeed_udc_ep_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 4,
    },
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

static void aspeed_udc_reset_hold(Object *obj, ResetType type)
{
    AspeedUDCState *s = ASPEED_UDC(obj);
    USBDevice *udev;
    int i;

    memset(s->regs, 0, sizeof(s->regs));
    for (i = 0; i < ASPEED_UDC_NUM_EP; i++) {
        memset(s->ep[i].regs, 0, sizeof(s->ep[i].regs));
        s->ep[i].pkt = NULL;
        s->ep[i].desc_off = 0;
    }

    /* Device-reset default: root, DMA and EP-pool soft-reset bits set */
    s->regs[R_UDC_DEV_RESET] = (R_UDC_DEV_RESET_ROOT_MASK |
                                R_UDC_DEV_RESET_DMA_MASK |
                                R_UDC_DEV_RESET_EP_POOL_MASK);
    s->ep0_packet = NULL;

    /*
     * A guest reboot resets the controller but leaves the USB device
     * attached to the host bus with no guest gadget driver behind it.
     * Detach it, otherwise the rebooted host fails to re-enumerate the
     * driverless gadget device; it re-attaches when the new driver asserts
     * pull-up.
     */
    if (s->usbgadget) {
        udev = USB_DEVICE(s->usbgadget);
        if (udev->attached) {
            usb_device_detach(udev);
        }
    }
}

static void aspeed_udc_realize(DeviceState *dev, Error **errp)
{
    SysBusDevice *sbd = SYS_BUS_DEVICE(dev);
    AspeedUDCState *s = ASPEED_UDC(dev);
    int i;

    if (!s->dram_mr) {
        error_setg(errp, TYPE_ASPEED_UDC ": 'dram' link not set");
        return;
    }
    address_space_init(&s->dram_as, s->dram_mr, "dram");

    memory_region_init(&s->udc_container, OBJECT(s), TYPE_ASPEED_UDC,
                       ASPEED_UDC_MEM_SIZE);
    memory_region_init_io(&s->root_mr, OBJECT(s), &aspeed_udc_ops, s,
                          TYPE_ASPEED_UDC ".root",
                          ASPEED_UDC_ROOT_NR_REGS << 2);
    memory_region_add_subregion(&s->udc_container, 0, &s->root_mr);

    /* Each programmable endpoint has its own register bank */
    for (i = 0; i < ASPEED_UDC_NUM_EP; i++) {
        g_autofree char *name = g_strdup_printf(TYPE_ASPEED_UDC ".ep%d", i);

        s->ep[i].index = i;
        memory_region_init_io(&s->ep[i].mr, OBJECT(s), &aspeed_udc_ep_ops,
                              &s->ep[i], name, ASPEED_UDC_EP_NR_REGS << 2);
        memory_region_add_subregion(&s->udc_container,
                                    ASPEED_UDC_EP_REG_BASE +
                                    i * (ASPEED_UDC_EP_NR_REGS << 2),
                                    &s->ep[i].mr);
    }

    sysbus_init_mmio(sbd, &s->udc_container);
    sysbus_init_irq(sbd, &s->irq);
}

static const Property aspeed_udc_properties[] = {
    DEFINE_PROP_LINK("dram", AspeedUDCState, dram_mr,
                     TYPE_MEMORY_REGION, MemoryRegion *),
};

static void aspeed_udc_init(Object *obj)
{
    AspeedUDCState *s = ASPEED_UDC(obj);

    object_property_add_link(obj, "usbgadget", TYPE_ASPEED_UDC_GADGET,
                             (Object **)&s->usbgadget,
                             object_property_allow_set_link, 0);
}

static void aspeed_udc_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    ResettableClass *rc = RESETTABLE_CLASS(klass);

    dc->desc = "ASPEED USB Device Controller";
    dc->realize = aspeed_udc_realize;
    rc->phases.hold = aspeed_udc_reset_hold;
    device_class_set_props(dc, aspeed_udc_properties);
}

/*
 * USB device: gadget device presented on a host controller's bus
 *
 * These callbacks run in the context of the host controller. They translate
 * host transactions into the controller interrupts/state the guest gadget
 * driver expects, then defer (USB_RET_ASYNC) until the driver responds
 * through the MMIO register interface above.
 */

static int aspeed_udc_find_ep(AspeedUDCState *s, int ep_nr, bool is_out)
{
    uint32_t cfg;
    int i;

    for (i = 0; i < ASPEED_UDC_NUM_EP; i++) {
        cfg = s->ep[i].regs[R_EP_CONFIG];

        if (!FIELD_EX32(cfg, EP_CONFIG, ENABLE) ||
            FIELD_EX32(cfg, EP_CONFIG, EP_NUM) != ep_nr) {
            continue;
        }
        if (FIELD_EX32(cfg, EP_CONFIG, DIR_OUT) == is_out) {
            return i;
        }
    }

    return -1;
}

static void aspeed_udc_ep_data_in(AspeedUDCState *s, int ep, USBPacket *p)
{
    AspeedUDCEP *e = &s->ep[ep];
    uint32_t rptr = FIELD_EX32(e->regs[R_EP_DMA_STS], EP_DMA_STS, RPTR);
    uint32_t wptr = FIELD_EX32(e->regs[R_EP_DMA_STS], EP_DMA_STS, WPTR);

    if (rptr == wptr) {
        /*
         * No IN data is queued yet. Save the packet and return ASYNC
         * instead of NAK. A NAK would make the host retry slowly.
         * aspeed_udc_ep_in_kick() serves and completes this packet later,
         * once the guest gadget driver queues descriptors.
         */
        e->pkt = p;
        p->status = USB_RET_ASYNC;
        return;
    }

    switch (aspeed_udc_ep_xfer_in(s, ep, p)) {
    case ASPEED_UDC_XFER_DONE:
        p->status = USB_RET_SUCCESS;
        break;
    case ASPEED_UDC_XFER_MORE:
        /* not fully sent yet: save the packet, wait for more descriptors */
        e->pkt = p;
        p->status = USB_RET_ASYNC;
        break;
    case ASPEED_UDC_XFER_ERROR:
        p->status = USB_RET_IOERROR;
        break;
    }
}

static void aspeed_udc_ep_data_out(AspeedUDCState *s, int ep, USBPacket *p)
{
    AspeedUDCEP *e = &s->ep[ep];
    uint32_t sts = e->regs[R_EP_DMA_STS];

    trace_aspeed_udc_ep_data_out(ep, FIELD_EX32(sts, EP_DMA_STS, WPTR),
                                 FIELD_EX32(sts, EP_DMA_STS, PKT_SIZE),
                                 p->iov.size);
    if (!FIELD_EX32(sts, EP_DMA_STS, WPTR)) {
        /*
         * No OUT buffer is ready yet. Save the packet and return ASYNC
         * instead of NAK. Writing now could use an old buffer address and
         * lose the data (for example a mass-storage CBW). A NAK would make
         * the host retry slowly. aspeed_udc_ep_out_kick() delivers this
         * packet later, once the guest gadget driver sets up a buffer.
         */
        e->pkt = p;
        p->status = USB_RET_ASYNC;
        return;
    }

    switch (aspeed_udc_ep_xfer_out(s, ep, p)) {
    case ASPEED_UDC_XFER_DONE:
        p->status = USB_RET_SUCCESS;
        break;
    case ASPEED_UDC_XFER_MORE:
        /* not fully received yet: save the packet, wait for the next buffer */
        e->pkt = p;
        p->status = USB_RET_ASYNC;
        break;
    case ASPEED_UDC_XFER_ERROR:
        p->status = USB_RET_IOERROR;
        break;
    }
}

static void aspeed_udc_gadget_handle_reset(USBDevice *udev)
{
    AspeedUDCState *s = ASPEED_UDC_GADGET(udev)->udc;

    s->ep0_packet = NULL;
    s->ep0_offset = 0;
    /* The EHCI host is High-Speed; advertise it to the guest gadget driver */
    s->regs[R_UDC_STS] = R_UDC_STS_HIGHSPEED_MASK;
    trace_aspeed_udc_reset(s->regs[R_UDC_IER]);
    aspeed_udc_raise_isr(s, R_UDC_ISR_BUS_RESET_MASK);
}

static void aspeed_udc_gadget_handle_control(USBDevice *udev, USBPacket *p,
                                          int request, int value, int index,
                                          int length, uint8_t *data)
{
    AspeedUDCState *s = ASPEED_UDC_GADGET(udev)->udc;
    uint8_t req = request & 0xff;
    uint8_t type = request >> 8;

    /*
     * Reconstruct the 8-byte SETUP packet into the SETUP data buffer where
     * the guest gadget driver reads it from.
     */
    s->regs[R_UDC_SETUP0] = type | (req << 8) | ((value & 0xffff) << 16);
    s->regs[R_UDC_SETUP1] = (index & 0xffff) | ((length & 0xffff) << 16);

    /* A new SETUP clears the EP0 STALL condition */
    s->regs[R_UDC_EP0_CTRL] &= ~R_UDC_EP0_CTRL_STALL_MASK;

    s->ep0_packet = p;
    s->ep0_data = data;
    s->ep0_setup_len = length;
    s->ep0_offset = 0;
    s->ep0_dir_in = (type & USB_DIR_IN);

    trace_aspeed_udc_ep0_setup(type, req, value, index, length,
                               s->ep0_dir_in, udev->addr);

    /*
     * SET_ADDRESS is delivered while the device still answers at the default
     * address 0 and carries the new address in wValue. The host controller
     * keeps this transfer's queue bound to address 0 until it completes, so
     * apply the new address synchronously as the transfer completes.
     * Completing it asynchronously (USB_RET_ASYNC) would change udev->addr
     * while the queue is still bound to 0; the host controller sees the
     * mismatch, tears the queue down and enumeration breaks. The guest gadget
     * driver is still notified so its state machine advances.
     */
    if (type == 0 && req == USB_REQ_SET_ADDRESS) {
        udev->addr = value;
        s->ep0_packet = NULL;
        aspeed_udc_raise_isr(s, R_UDC_ISR_EP0_SETUP_MASK);
        p->status = USB_RET_SUCCESS;
        return;
    }

    aspeed_udc_raise_isr(s, R_UDC_ISR_EP0_SETUP_MASK);
    p->status = USB_RET_ASYNC;
}

static void aspeed_udc_gadget_handle_data(USBDevice *udev, USBPacket *p)
{
    AspeedUDCState *s = ASPEED_UDC_GADGET(udev)->udc;
    bool is_out = (p->pid == USB_TOKEN_OUT);
    int ep = aspeed_udc_find_ep(s, p->ep->nr, is_out);

    trace_aspeed_udc_handle_data(p->ep->nr, is_out ? "OUT" : "IN",
                                 p->iov.size, ep);
    if (ep < 0) {
        p->status = USB_RET_STALL;
        return;
    }

    if (is_out) {
        aspeed_udc_ep_data_out(s, ep, p);
    } else {
        aspeed_udc_ep_data_in(s, ep, p);
    }
}

static void aspeed_udc_gadget_cancel_packet(USBDevice *udev, USBPacket *p)
{
    AspeedUDCState *s = ASPEED_UDC_GADGET(udev)->udc;
    int i;

    if (s->ep0_packet == p) {
        s->ep0_packet = NULL;
    }
    for (i = 0; i < ASPEED_UDC_NUM_EP; i++) {
        if (s->ep[i].pkt == p) {
            s->ep[i].pkt = NULL;
        }
    }
}

static void aspeed_udc_gadget_realize(USBDevice *udev, Error **errp)
{
    AspeedUDCGadget *dev = ASPEED_UDC_GADGET(udev);

    if (!dev->udc) {
        error_setg(errp, TYPE_ASPEED_UDC_GADGET ": 'udc' link is not set");
        return;
    }
    /* Bind this gadget to its controller through the link property */
    object_property_set_link(OBJECT(dev->udc), "usbgadget", OBJECT(dev),
                             &error_abort);

    udev->auto_attach = 0;
    /* The ASPEED UDC is USB 2.0, so it only runs at High-Speed for now */
    udev->speed = USB_SPEED_HIGH;
    udev->speedmask = USB_SPEED_MASK_HIGH;
}

static void aspeed_udc_gadget_unrealize(USBDevice *udev)
{
    AspeedUDCGadget *dev = ASPEED_UDC_GADGET(udev);

    if (dev->udc && dev->udc->usbgadget == dev) {
        object_property_set_link(OBJECT(dev->udc), "usbgadget", NULL,
                                 &error_abort);
    }
}

static const Property aspeed_udc_gadget_props[] = {
    DEFINE_PROP_LINK("udc", AspeedUDCGadget, udc, TYPE_ASPEED_UDC,
                     AspeedUDCState *),
};

static void aspeed_udc_gadget_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    USBDeviceClass *uc = USB_DEVICE_CLASS(klass);

    dc->desc           = "ASPEED UDC gadget device";
    uc->product_desc   = "ASPEED UDC gadget";
    uc->realize        = aspeed_udc_gadget_realize;
    uc->unrealize      = aspeed_udc_gadget_unrealize;
    uc->handle_reset   = aspeed_udc_gadget_handle_reset;
    uc->handle_control = aspeed_udc_gadget_handle_control;
    uc->handle_data    = aspeed_udc_gadget_handle_data;
    uc->cancel_packet  = aspeed_udc_gadget_cancel_packet;
    device_class_set_props(dc, aspeed_udc_gadget_props);
}

static const TypeInfo aspeed_udc_types[] = {
    {
        .name          = TYPE_ASPEED_UDC,
        .parent        = TYPE_SYS_BUS_DEVICE,
        .instance_size = sizeof(AspeedUDCState),
        .instance_init = aspeed_udc_init,
        .class_init    = aspeed_udc_class_init,
    },
    {
        .name          = TYPE_ASPEED_UDC_GADGET,
        .parent        = TYPE_USB_DEVICE,
        .instance_size = sizeof(AspeedUDCGadget),
        .class_init    = aspeed_udc_gadget_class_init,
    },
};

DEFINE_TYPES(aspeed_udc_types)
