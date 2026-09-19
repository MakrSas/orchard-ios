/*
 * ASPEED ACRY Engine
 *
 * Copyright (C) 2026 ASPEED Technology Inc.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * The datasheet documents the ACRY engine as supporting both RSA and
 * ECDSA, but ECDSA is broken on this hardware, so only RSA is modelled
 * here.
 */

#include "qemu/osdep.h"
#include "qemu/cutils.h"
#include "qemu/log.h"
#include "hw/misc/aspeed_acry.h"
#include "hw/core/qdev-properties.h"
#include "hw/core/irq.h"
#include "hw/core/registerfields.h"
#include "qapi/error.h"
#include "crypto/akcipher.h"
#include "crypto/der.h"
#include "trace.h"

REG32(ACRY_TRIGGER, 0x000)
    FIELD(ACRY_TRIGGER, RSA_START, 0, 1)
REG32(ACRY_DMA_SRC, 0x04C)
REG32(ACRY_DMA_LEN, 0x050)
    FIELD(ACRY_DMA_LEN, DATALEN, 0, 16)
REG32(ACRY_INT_MASK, 0x3F8)
    FIELD(ACRY_INT_MASK, RSA_DMA_MASK, 2, 1)
    FIELD(ACRY_INT_MASK, RSA_ENG_MASK, 1, 1)
REG32(ACRY_STATUS, 0x3FC)
    FIELD(ACRY_STATUS, RSA_DMA_DONE, 2, 1)
    FIELD(ACRY_STATUS, RSA_ENG_DONE, 1, 1)

/*
 * Total size of the interleaved buffer. Data is 4 of every 12
 * dwords of a block, one third of the buffer, so the whole buffer is 3x
 * the data region.
 */
#define ASPEED_ACRY_SRAM_SIZE   (3 * ASPEED_ACRY_DATA_MAX_LEN)

#define ASPEED_ACRY_BYTES_PER_DWORD  4
#define ASPEED_ACRY_LANES_PER_BLOCK  4
/* Each block holds 3 regions (exp, mod, data) of LANES_PER_BLOCK dwords. */
#define ASPEED_ACRY_DWORDS_PER_BLOCK (3 * ASPEED_ACRY_LANES_PER_BLOCK)

/* Dwords into each block where each operand's region starts. */
#define ASPEED_ACRY_EXP_OFFSET   (0 * ASPEED_ACRY_LANES_PER_BLOCK)
#define ASPEED_ACRY_MOD_OFFSET   (1 * ASPEED_ACRY_LANES_PER_BLOCK)
#define ASPEED_ACRY_DATA_OFFSET  (2 * ASPEED_ACRY_LANES_PER_BLOCK)

static void aspeed_acry_hexdump(const char *desc, const uint8_t *buf,
                                size_t size)
{
    g_autoptr(GString) str = g_string_sized_new(64);
    size_t len;
    size_t i;

    for (i = 0; i < size; i += len) {
        len = MIN(16, size - i);
        g_string_truncate(str, 0);
        qemu_hexdump_line(str, buf + i, len, 1, 4);
        trace_aspeed_acry_hexdump(desc, i, str->str);
    }
}

/*
 * The interleaved buffer is a series of 12-dword blocks, each split into
 * three 4-dword regions - exp, mod, data:
 *
 *   dword in block:  0    1    2    3    4    5    6    7    8    9   10   11
 *   region:          \---- exp ----/    \---- mod ----/    \---- data ----/
 *   lane:            0    1    2    3    0    1    2    3    0    1    2    3
 *
 * Successive blocks hold the next 4 dwords of each operand, so operand
 * dword d is in block (d / 4), lane (d % 4). Dwords are little-endian, so
 * byte b of dword D is at byte D * 4 + b.
 *
 * Return the buffer offset of byte 'op_byte' (op_byte = 0 = least
 * significant) of the operand whose region starts 'region' dwords into
 * each block (0 = exp, 4 = mod, 8 = data).
 */
static int aspeed_acry_operand_offset(int region, int op_byte)
{
    int byte_in_dword;
    int op_dword;
    int offset;
    int block;
    int lane;

    op_dword = op_byte / ASPEED_ACRY_BYTES_PER_DWORD;
    byte_in_dword = op_byte % ASPEED_ACRY_BYTES_PER_DWORD;
    block = op_dword / ASPEED_ACRY_LANES_PER_BLOCK;
    lane = op_dword % ASPEED_ACRY_LANES_PER_BLOCK;

    offset = (block * ASPEED_ACRY_DWORDS_PER_BLOCK + region + lane)
             * ASPEED_ACRY_BYTES_PER_DWORD + byte_in_dword;
    assert(offset < ASPEED_ACRY_SRAM_SIZE);

    return offset;
}

/*
 * Read one operand out of the buffer as a big-endian magnitude.
 *
 * The operand's bytes are scattered through buf; byte k (significance level
 * k, k = 0 = least significant) is at aspeed_acry_operand_offset(region, k).
 * Walk from the top down, drop leading zero bytes, and write the result most
 * significant byte first into out[]. Returns the number of bytes written
 * (the value 0 yields a single 0x00 byte, so always >= 1).
 */
static int aspeed_acry_extract_be(const uint8_t *buf, int region,
                                  int max_bytes, uint8_t *out)
{
    int offset;
    int msb;
    int len;
    int k;

    /* Highest significance level holding a non-zero byte (skip leading 0s). */
    for (msb = max_bytes - 1; msb >= 0; msb--) {
        offset = aspeed_acry_operand_offset(region, msb);
        if (buf[offset] != 0) {
            break;
        }
    }

    /* All bytes zero: the value is 0. */
    if (msb < 0) {
        out[0] = 0;
        return 1;
    }

    /* Copy most significant byte first: level msb down to level 0. */
    len = 0;
    for (k = msb; k >= 0; k--) {
        offset = aspeed_acry_operand_offset(region, k);
        out[len++] = buf[offset];
    }

    return len;
}

/*
 * Return a DER INTEGER body for the unsigned big-endian magnitude 'be'.
 *
 * DER INTEGERs are signed, so if the top byte has bit 7 set the value
 * would decode as negative; prepend a 0x00 guard byte in that case.
 *
 * The padded copy is written into 'pad_buf' (caller-owned, sized len + 1)
 * rather than a local, because qcrypto_der_encode_int() only stores the
 * pointer we hand it - the bytes are not copied until
 * qcrypto_der_encode_ctx_flush_and_free() - so the body must stay valid
 * until then. Returns a pointer into 'be' or 'pad_buf' as appropriate,
 * with the body length in *body_len.
 */
static const uint8_t *aspeed_acry_der_uint_body(const uint8_t *be, size_t len,
                                                uint8_t *pad_buf,
                                                size_t *body_len)
{
    if (be[0] & 0x80) {
        pad_buf[0] = 0x00;
        memcpy(pad_buf + 1, be, len);
        *body_len = len + 1;
        return pad_buf;
    }

    *body_len = len;
    return be;
}

/*
 * DER-encode a "RsaPubKey ::= SEQUENCE { n INTEGER, e INTEGER }" (see
 * crypto/rsakey.h), the format expected by qcrypto_akcipher_new(). n and e
 * are minimal big-endian magnitudes (as produced by
 * aspeed_acry_extract_be()); the engine does a raw modexp, so the guest's
 * exponent is always encoded here as the public 'e'.
 */
static uint8_t *aspeed_acry_der_encode_pubkey(const uint8_t *n, size_t n_len,
                                              const uint8_t *e, size_t e_len,
                                              size_t *out_len)
{
    QCryptoEncodeContext *ctx = qcrypto_der_encode_ctx_new();
    uint8_t n_pad[ASPEED_ACRY_MAX_BYTES + 1];
    uint8_t e_pad[ASPEED_ACRY_MAX_BYTES + 1];
    const uint8_t *n_body;
    const uint8_t *e_body;
    size_t n_body_len;
    size_t e_body_len;
    uint8_t *buf;

    n_body = aspeed_acry_der_uint_body(n, n_len, n_pad, &n_body_len);
    e_body = aspeed_acry_der_uint_body(e, e_len, e_pad, &e_body_len);

    qcrypto_der_encode_seq_begin(ctx);
    qcrypto_der_encode_int(ctx, n_body, n_body_len);
    qcrypto_der_encode_int(ctx, e_body, e_body_len);
    qcrypto_der_encode_seq_end(ctx);

    *out_len = qcrypto_der_encode_ctx_buffer_len(ctx);
    buf = g_malloc(*out_len);
    qcrypto_der_encode_ctx_flush_and_free(ctx, buf);

    return buf;
}

/*
 * Store the RSA result into the output SRAM data region as 'n_len' bytes
 * (the key size): the low 'result_len' bytes are result_be (big-endian),
 * the rest is zero. sram_as is a 0-based AddressSpace over the SRAM, so the
 * offset from aspeed_acry_operand_offset() is used directly; each data dword
 * is written as a little-endian word.
 */
static bool aspeed_acry_store_result(AspeedACRYState *s,
                                     const uint8_t *result_be,
                                     int result_len, int n_len, Error **errp)
{
    uint32_t result_word;
    MemTxResult res;
    int offset;
    int src;
    int i;
    int j;

    /* result_be is MSB-first; take bytes from its LSB end. */
    src = result_len - 1;
    for (i = 0; i < n_len / ASPEED_ACRY_BYTES_PER_DWORD; i++) {
        /* Pack up to 4 result bytes (LSB first) into a little-endian dword. */
        result_word = 0;
        for (j = 0; j < ASPEED_ACRY_BYTES_PER_DWORD; j++) {
            if (src >= 0) {
                result_word |= (uint32_t)result_be[src--] << (8 * j);
            }
        }

        offset = aspeed_acry_operand_offset(ASPEED_ACRY_DATA_OFFSET,
                                            ASPEED_ACRY_BYTES_PER_DWORD * i);
        address_space_stl_le(&s->sram_as, offset, result_word,
                             MEMTXATTRS_UNSPECIFIED, &res);
        if (res != MEMTX_OK) {
            error_setg(errp, "failed to write result to SRAM");
            return false;
        }
    }

    return true;
}

static void aspeed_acry_clear_result(AspeedACRYState *s)
{
    MemTxResult res;
    int offset;
    int i;

    for (i = 0; i < ASPEED_ACRY_MAX_BYTES / ASPEED_ACRY_BYTES_PER_DWORD; i++) {
        offset = aspeed_acry_operand_offset(ASPEED_ACRY_DATA_OFFSET,
                                            ASPEED_ACRY_BYTES_PER_DWORD * i);
        address_space_stl_le(&s->sram_as, offset, 0,
                             MEMTXATTRS_UNSPECIFIED, &res);
        if (res != MEMTX_OK) {
            qemu_log_mask(LOG_GUEST_ERROR, "%s: failed to clear result\n",
                          __func__);
            return;
        }
    }
}

static bool aspeed_acry_do_rsa(AspeedACRYState *s, Error **errp)
{
    QCryptoAkCipherOptions opts = {
        .alg = QCRYPTO_AK_CIPHER_ALGO_RSA,
        .u.rsa = {
            .padding_alg = QCRYPTO_RSA_PADDING_ALGO_RAW,
        },
    };
    uint32_t len = FIELD_EX32(s->regs[R_ACRY_DMA_LEN], ACRY_DMA_LEN, DATALEN);
    g_autofree uint8_t *src_buf = g_malloc0(ASPEED_ACRY_SRAM_SIZE);
    g_autofree uint8_t *data = g_malloc0(ASPEED_ACRY_DATA_MAX_LEN);
    uint8_t result[ASPEED_ACRY_MAX_BYTES] = { 0 };
    uint64_t src_addr = s->regs[R_ACRY_DMA_SRC];
    g_autofree uint8_t *der_key = NULL;
    uint8_t n[ASPEED_ACRY_MAX_BYTES];
    uint8_t e[ASPEED_ACRY_MAX_BYTES];
    QCryptoAkCipher *cipher = NULL;
    int result_len = 0;
    size_t der_len;
    int data_len;
    int n_len;
    int e_len;

    if (!qcrypto_akcipher_supports(&opts)) {
        error_setg(errp, "RSA ModExp not supported by the crypto backend");
        return false;
    }

    if (len == 0 || len > ASPEED_ACRY_SRAM_SIZE) {
        error_setg(errp, "invalid DMA length %u", len);
        return false;
    }

    trace_aspeed_acry_rsa_trigger(src_addr, len);

    if (address_space_read(&s->dram_as, src_addr, MEMTXATTRS_UNSPECIFIED,
                           src_buf, len) != MEMTX_OK) {
        error_setg(errp, "failed to read DMA buffer at 0x%" PRIx64, src_addr);
        return false;
    }

    n_len = aspeed_acry_extract_be(src_buf, ASPEED_ACRY_MOD_OFFSET,
                                   ASPEED_ACRY_MAX_BYTES, n);
    e_len = aspeed_acry_extract_be(src_buf, ASPEED_ACRY_EXP_OFFSET,
                                   ASPEED_ACRY_MAX_BYTES, e);
    data_len = aspeed_acry_extract_be(src_buf, ASPEED_ACRY_DATA_OFFSET,
                                      ASPEED_ACRY_DATA_MAX_LEN, data);

    if (trace_event_get_state_backends(TRACE_ASPEED_ACRY_HEXDUMP)) {
        aspeed_acry_hexdump("buf", src_buf, len);
        aspeed_acry_hexdump("n", n, n_len);
        aspeed_acry_hexdump("e", e, e_len);
        aspeed_acry_hexdump("data", data, data_len);
    }

    der_key = aspeed_acry_der_encode_pubkey(n, n_len, e, e_len, &der_len);
    cipher = qcrypto_akcipher_new(&opts, QCRYPTO_AK_CIPHER_KEY_TYPE_PUBLIC,
                                  der_key, der_len, errp);
    if (!cipher) {
        error_prepend(errp, "failed to create RSA cipher: ");
        return false;
    }

    result_len = qcrypto_akcipher_encrypt(cipher, data, data_len,
                                          result, sizeof(result), errp);
    qcrypto_akcipher_free(cipher);
    if (result_len < 0) {
        error_prepend(errp, "RSA modexp failed: ");
        return false;
    }

    if (!aspeed_acry_store_result(s, result, result_len, n_len, errp)) {
        return false;
    }

    if (trace_event_get_state_backends(TRACE_ASPEED_ACRY_HEXDUMP)) {
        aspeed_acry_hexdump("result", result, result_len);
    }

    return true;
}

static uint64_t aspeed_acry_read(void *opaque, hwaddr offset, unsigned int size)
{
    AspeedACRYState *s = ASPEED_ACRY(opaque);
    uint32_t reg = offset >> 2;

    trace_aspeed_acry_read(offset, s->regs[reg]);

    return s->regs[reg];
}

static void aspeed_acry_write(void *opaque, hwaddr offset, uint64_t data,
                              unsigned int size)
{
    AspeedACRYState *s = ASPEED_ACRY(opaque);
    uint32_t reg = offset >> 2;
    Error *local_err = NULL;

    trace_aspeed_acry_write(offset, data);

    switch (reg) {
    case R_ACRY_DMA_SRC:
        /*
         * The DMA source register holds a CPU-visible DRAM address (e.g.
         * 0x8xxxxxxx on AST2600); the engine addresses DRAM from offset 0,
         * so mask off the top bit to get the DRAM-relative offset.
         */
        data &= 0x7FFFFFFF;
        break;
    case R_ACRY_STATUS:
        data = s->regs[R_ACRY_STATUS] & ~data;
        if (!(data & (R_ACRY_STATUS_RSA_ENG_DONE_MASK |
                      R_ACRY_STATUS_RSA_DMA_DONE_MASK))) {
            qemu_irq_lower(s->irq);
        }
        break;
    case R_ACRY_TRIGGER:
        if (FIELD_EX32(data, ACRY_TRIGGER, RSA_START)) {
            if (!aspeed_acry_do_rsa(s, &local_err)) {
                qemu_log_mask(LOG_GUEST_ERROR, "%s: %s\n", __func__,
                              error_get_pretty(local_err));
                error_free(local_err);
                aspeed_acry_clear_result(s);
            }

            s->regs[R_ACRY_STATUS] |= R_ACRY_STATUS_RSA_ENG_DONE_MASK |
                                      R_ACRY_STATUS_RSA_DMA_DONE_MASK;
            if (s->regs[R_ACRY_INT_MASK] &
                (R_ACRY_INT_MASK_RSA_ENG_MASK_MASK |
                 R_ACRY_INT_MASK_RSA_DMA_MASK_MASK)) {
                qemu_irq_raise(s->irq);
            }
        }
        break;
    default:
        break;
    }

    s->regs[reg] = data;
}

static const MemoryRegionOps aspeed_acry_ops = {
    .read = aspeed_acry_read,
    .write = aspeed_acry_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 4,
    },
};

static void aspeed_acry_reset_hold(Object *obj, ResetType type)
{
    AspeedACRYState *s = ASPEED_ACRY(obj);

    memset(s->regs, 0, sizeof(s->regs));
}

static void aspeed_acry_realize(DeviceState *dev, Error **errp)
{
    SysBusDevice *sbd = SYS_BUS_DEVICE(dev);
    AspeedACRYState *s = ASPEED_ACRY(dev);

    if (!s->dram_mr) {
        error_setg(errp, TYPE_ASPEED_ACRY ": 'dram' link not set");
        return;
    }

    if (!s->sram_mr) {
        error_setg(errp, TYPE_ASPEED_ACRY ": 'sram' link not set");
        return;
    }

    address_space_init(&s->dram_as, s->dram_mr, "dram");
    address_space_init(&s->sram_as, s->sram_mr, "sram");

    memory_region_init_io(&s->iomem, OBJECT(s), &aspeed_acry_ops, s,
                          TYPE_ASPEED_ACRY, ASPEED_ACRY_NR_REGS << 2);
    sysbus_init_mmio(sbd, &s->iomem);

    sysbus_init_irq(sbd, &s->irq);
}

static const Property aspeed_acry_properties[] = {
    DEFINE_PROP_LINK("dram", AspeedACRYState, dram_mr,
                     TYPE_MEMORY_REGION, MemoryRegion *),
    DEFINE_PROP_LINK("sram", AspeedACRYState, sram_mr,
                     TYPE_MEMORY_REGION, MemoryRegion *),
};

static void aspeed_acry_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    ResettableClass *rc = RESETTABLE_CLASS(klass);

    dc->desc = "ASPEED ACRY Engine";
    dc->realize = aspeed_acry_realize;
    rc->phases.hold = aspeed_acry_reset_hold;
    device_class_set_props(dc, aspeed_acry_properties);
}

static const TypeInfo aspeed_acry_types[] = {
    {
        .name = TYPE_ASPEED_ACRY,
        .parent = TYPE_SYS_BUS_DEVICE,
        .instance_size = sizeof(AspeedACRYState),
        .class_init = aspeed_acry_class_init,
    },
};

DEFINE_TYPES(aspeed_acry_types)
