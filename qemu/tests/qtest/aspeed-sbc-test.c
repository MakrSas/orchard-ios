/*
 * QTest testcase for the ASPEED Secure Boot Controller (SBC)
 *
 * Copyright (C) 2026 ASPEED Technology Inc.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "libqtest.h"
#include "qemu/bitops.h"
#include "crypto/akcipher.h"

/* SBC register block */
#define SBC_STATUS              0x014
#define  SBC_ECDSA_VERIFY_PASS  BIT(21)
#define  SBC_ECDSA_VERIFY_DONE  BIT(20)
#define SBC_SEC_TRIGGER         0x0bc
#define  SBC_ECDSA_CMD_TRIGGER  BIT(1)

/*
 * SEC SRAM operand offsets for a secp384r1 ECDSA verify. Every operand is a
 * 48-byte big-endian integer.
 */
#define ECDSA_SRAM_QX   0x2080
#define ECDSA_SRAM_QY   0x20c0
#define ECDSA_SRAM_R    0x21c0
#define ECDSA_SRAM_S    0x2200
#define ECDSA_SRAM_M    0x2240

/*
 * ECDSA secp384r1 / SHA-384 known-answer vector from the Linux kernel crypto
 * self-test manager (ecdsa_nist_p384_tv_template, sha384 entry in
 * crypto/testmgr.h, v6.18), decoded into raw big-endian form: public key
 * Qx || Qy, signature r || s and the SHA-384 message digest.
 *
 *   https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/crypto/testmgr.h?h=v6.18
 */
static const uint8_t ecdsa_p384_pubkey[96] = {
    /* Qx */
    0x3a, 0x2f, 0x62, 0xe7, 0x1a, 0xcf, 0x24, 0xd0,
    0x0b, 0x7c, 0xe0, 0xed, 0x46, 0x0a, 0x4f, 0x74,
    0x16, 0x43, 0xe9, 0x1a, 0x25, 0x7c, 0x55, 0xff,
    0xf0, 0x29, 0x68, 0x66, 0x20, 0x91, 0xf9, 0xdb,
    0x2b, 0xf6, 0xb3, 0x6c, 0x54, 0x01, 0xca, 0xc7,
    0x6a, 0x5c, 0x0d, 0xeb, 0x68, 0xd9, 0x3c, 0xf1,
    /* Qy */
    0x01, 0x74, 0x1f, 0xf9, 0x6c, 0xe5, 0x5b, 0x60,
    0xe9, 0x7f, 0x5d, 0xb3, 0x12, 0x80, 0x2a, 0xd8,
    0x67, 0x92, 0xc9, 0x0e, 0x4c, 0x4c, 0x6b, 0xa1,
    0xb2, 0xa8, 0x1e, 0xac, 0x1c, 0x97, 0xd9, 0x21,
    0x67, 0xe5, 0x1b, 0x5a, 0x52, 0x31, 0x68, 0xd6,
    0xee, 0xf0, 0x19, 0xb0, 0x55, 0xed, 0x89, 0x9e,
};

static const uint8_t ecdsa_p384_signature[96] = {
    /* r */
    0x9b, 0x28, 0x68, 0xc0, 0xa1, 0xea, 0x8c, 0x50,
    0xee, 0x2e, 0x62, 0x35, 0x46, 0xfa, 0x00, 0xd8,
    0x2d, 0x7a, 0x91, 0x5f, 0x49, 0x2d, 0x22, 0x08,
    0x29, 0xe6, 0xfb, 0xca, 0x8c, 0xd6, 0xb6, 0xb4,
    0x3b, 0x1f, 0x07, 0x8f, 0x15, 0x02, 0xfe, 0x1d,
    0xa2, 0xa4, 0xc8, 0xf2, 0xea, 0x9d, 0x11, 0x1f,
    /* s */
    0xfc, 0x50, 0xf6, 0x43, 0xbd, 0x50, 0x82, 0x0e,
    0xbf, 0xe3, 0x75, 0x24, 0x49, 0xac, 0xfb, 0xc8,
    0x71, 0xcd, 0x8f, 0x18, 0x99, 0xf0, 0x0f, 0x13,
    0x44, 0x92, 0x8c, 0x86, 0x99, 0x65, 0xb3, 0x97,
    0x96, 0x17, 0x04, 0xc9, 0x05, 0x77, 0xf1, 0x8e,
    0xab, 0x8d, 0x4e, 0xde, 0xe6, 0x6d, 0x9b, 0x66,
};

static const uint8_t ecdsa_p384_dgst[48] = {
    0x8d, 0xf2, 0xc0, 0xe9, 0xa8, 0xf3, 0x8e, 0x44,
    0xc4, 0x8c, 0x1a, 0xa0, 0xb8, 0xd7, 0x17, 0xdf,
    0xf2, 0x37, 0x1b, 0xc6, 0xe3, 0xf5, 0x62, 0xcc,
    0x68, 0xf5, 0xd5, 0x0b, 0xbf, 0x73, 0x2b, 0xb1,
    0xb0, 0x4c, 0x04, 0x00, 0x31, 0xab, 0xfe, 0xc8,
    0xd6, 0x09, 0xc8, 0xf2, 0xea, 0xd3, 0x28, 0xff,
};

typedef struct AspeedSBCECDSA {
    const char *name;
    QCryptoCurveID curve_id;
    const uint8_t *pubkey;
    const uint8_t *signature;
    const uint8_t *dgst;
    size_t coord_len;
} AspeedSBCECDSA;

static const AspeedSBCECDSA sbc_ecdsa_tests[] = {
    {
        .name = "secp384r1",
        .curve_id = QCRYPTO_CURVE_ID_SECP384R1,
        .pubkey = ecdsa_p384_pubkey,
        .signature = ecdsa_p384_signature,
        .dgst = ecdsa_p384_dgst,
        .coord_len = 48,
    },
};

typedef struct AspeedSBCTest {
    const char *machine;
    uint32_t sec_addr;
    uint64_t sram_addr;
    int index;
} AspeedSBCTest;

static void sbc_ecdsa_stage(QTestState *qts, const AspeedSBCTest *c,
                            const AspeedSBCECDSA *t)
{
    uint32_t len = t->coord_len;

    qtest_memwrite(qts, c->sram_addr + ECDSA_SRAM_QX, t->pubkey, len);
    qtest_memwrite(qts, c->sram_addr + ECDSA_SRAM_QY, t->pubkey + len, len);
    qtest_memwrite(qts, c->sram_addr + ECDSA_SRAM_R, t->signature, len);
    qtest_memwrite(qts, c->sram_addr + ECDSA_SRAM_S, t->signature + len, len);
    qtest_memwrite(qts, c->sram_addr + ECDSA_SRAM_M, t->dgst, len);
}

/*
 * Drive one ECDSA verify through the engine the way the firmware does:
 * stage the operands in the SEC SRAM, trigger the command register and read
 * back the done/pass bits in the status register.
 */
static void test_ecdsa_verify(const void *opaque)
{
    const AspeedSBCTest *c = opaque;
    const AspeedSBCECDSA *t = &sbc_ecdsa_tests[c->index];
    QCryptoAkCipherOptions opts = {
        .alg = QCRYPTO_AK_CIPHER_ALGO_ECDSA,
        .u.ecdsa.curve_id = t->curve_id,
    };
    QTestState *qts;
    uint32_t status;
    uint8_t bad;

    if (!qcrypto_akcipher_supports(&opts)) {
        g_test_skip("ECDSA is not supported by the crypto backend");
        return;
    }

    qts = qtest_init(c->machine);

    /* A valid signature verifies */
    sbc_ecdsa_stage(qts, c, t);
    qtest_writel(qts, c->sec_addr + SBC_SEC_TRIGGER, SBC_ECDSA_CMD_TRIGGER);
    status = qtest_readl(qts, c->sec_addr + SBC_STATUS);
    g_assert_cmphex(status & (SBC_ECDSA_VERIFY_DONE | SBC_ECDSA_VERIFY_PASS),
                    ==, SBC_ECDSA_VERIFY_DONE | SBC_ECDSA_VERIFY_PASS);

    /* A tampered signature must fail */
    bad = t->signature[0] ^ 0xff;
    qtest_memwrite(qts, c->sram_addr + ECDSA_SRAM_R, &bad, 1);
    qtest_writel(qts, c->sec_addr + SBC_SEC_TRIGGER, SBC_ECDSA_CMD_TRIGGER);
    status = qtest_readl(qts, c->sec_addr + SBC_STATUS);
    g_assert_cmphex(status & SBC_ECDSA_VERIFY_DONE, ==, SBC_ECDSA_VERIFY_DONE);
    g_assert_cmphex(status & SBC_ECDSA_VERIFY_PASS, ==, 0);

    qtest_quit(qts);
}

static void aspeed_add_sbc_ecdsa_tests(const char *prefix, const char *machine,
                                       uint32_t sec_addr, uint64_t sram_addr)
{
    int i;

    for (i = 0; i < ARRAY_SIZE(sbc_ecdsa_tests); i++) {
        g_autofree char *path = NULL;
        AspeedSBCTest *t;

        path = g_strdup_printf("%s/sbc/ecdsa/%s", prefix,
                               sbc_ecdsa_tests[i].name);
        t = g_new0(AspeedSBCTest, 1);
        t->machine = machine;
        t->sec_addr = sec_addr;
        t->sram_addr = sram_addr;
        t->index = i;
        qtest_add_data_func_full(path, t, test_ecdsa_verify, g_free);
    }
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    aspeed_add_sbc_ecdsa_tests("ast1030", "-machine ast1030-evb",
                               0x7e6f2000, 0x79000000);

    aspeed_add_sbc_ecdsa_tests("ast1060", "-machine ast1060-evb",
                               0x7e6f2000, 0x79000000);

    return g_test_run();
}
