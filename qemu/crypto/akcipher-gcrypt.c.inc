/*
 * QEMU Crypto akcipher algorithms
 *
 * Copyright (c) 2022 Bytedance
 * Author: lei he <helei.sig11@bytedance.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <gcrypt.h>

#include "qemu/osdep.h"
#include "qemu/host-utils.h"
#include "crypto/akcipher.h"
#include "crypto/random.h"
#include "qapi/error.h"
#include "system/cryptodev.h"
#include "rsakey.h"

typedef struct QCryptoGcryptRSA {
    QCryptoAkCipher akcipher;
    gcry_sexp_t key;
    QCryptoRSAPaddingAlgo padding_alg;
    QCryptoHashAlgo hash_alg;
} QCryptoGcryptRSA;

typedef struct QCryptoGcryptECDSA {
    QCryptoAkCipher akcipher;
    QCryptoAkCipherKeyType type;
    gcry_sexp_t key;
    QCryptoCurveID curve_id;
} QCryptoGcryptECDSA;

static void qcrypto_gcrypt_rsa_free(QCryptoAkCipher *akcipher)
{
    QCryptoGcryptRSA *rsa = (QCryptoGcryptRSA *)akcipher;
    if (!rsa) {
        return;
    }

    gcry_sexp_release(rsa->key);
    g_free(rsa);
}

static QCryptoGcryptRSA *qcrypto_gcrypt_rsa_new(
    const QCryptoAkCipherOptionsRSA *opt,
    QCryptoAkCipherKeyType type,
    const uint8_t *key,  size_t keylen,
    Error **errp);

static QCryptoGcryptECDSA *qcrypto_gcrypt_ecdsa_new(
    const QCryptoAkCipherOptionsECDSA *opt,
    QCryptoAkCipherKeyType type,
    const uint8_t *key,  size_t keylen,
    Error **errp);

QCryptoAkCipher *qcrypto_akcipher_new(const QCryptoAkCipherOptions *opts,
                                      QCryptoAkCipherKeyType type,
                                      const uint8_t *key, size_t keylen,
                                      Error **errp)
{
    switch (opts->alg) {
    case QCRYPTO_AK_CIPHER_ALGO_RSA:
        return (QCryptoAkCipher *)qcrypto_gcrypt_rsa_new(
            &opts->u.rsa, type, key, keylen, errp);

    case QCRYPTO_AK_CIPHER_ALGO_ECDSA:
        return (QCryptoAkCipher *)qcrypto_gcrypt_ecdsa_new(
            &opts->u.ecdsa, type, key, keylen, errp);

    default:
        error_setg(errp, "Unsupported algorithm: %u", opts->alg);
        return NULL;
    }

    return NULL;
}

static void qcrypto_gcrypt_set_rsa_size(QCryptoAkCipher *akcipher, gcry_mpi_t n)
{
    size_t key_size = (gcry_mpi_get_nbits(n) + 7) / 8;
    akcipher->max_plaintext_len = key_size;
    akcipher->max_ciphertext_len = key_size;
    akcipher->max_dgst_len = key_size;
    akcipher->max_signature_len = key_size;
}

static int qcrypto_gcrypt_parse_rsa_private_key(
    QCryptoGcryptRSA *rsa,
    const uint8_t *key, size_t keylen, Error **errp)
{
    g_autoptr(QCryptoAkCipherRSAKey) rsa_key = qcrypto_akcipher_rsakey_parse(
        QCRYPTO_AK_CIPHER_KEY_TYPE_PRIVATE, key, keylen, errp);
    gcry_mpi_t n = NULL, e = NULL, d = NULL, p = NULL, q = NULL, u = NULL;
    bool compute_mul_inv = false;
    int ret = -1;
    gcry_error_t err;

    if (!rsa_key) {
        return ret;
    }

    err = gcry_mpi_scan(&n, GCRYMPI_FMT_STD,
                        rsa_key->n.data, rsa_key->n.len, NULL);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to parse RSA parameter n: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    err = gcry_mpi_scan(&e, GCRYMPI_FMT_STD,
                        rsa_key->e.data, rsa_key->e.len, NULL);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to parse RSA parameter e: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    err = gcry_mpi_scan(&d, GCRYMPI_FMT_STD,
                        rsa_key->d.data, rsa_key->d.len, NULL);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to parse RSA parameter d: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    err = gcry_mpi_scan(&p, GCRYMPI_FMT_STD,
                        rsa_key->p.data, rsa_key->p.len, NULL);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to parse RSA parameter p: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    err = gcry_mpi_scan(&q, GCRYMPI_FMT_STD,
                        rsa_key->q.data, rsa_key->q.len, NULL);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to parse RSA parameter q: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    if (gcry_mpi_cmp_ui(p, 0) > 0 && gcry_mpi_cmp_ui(q, 0) > 0) {
        compute_mul_inv = true;

        u = gcry_mpi_new(0);
        if (gcry_mpi_cmp(p, q) > 0) {
            gcry_mpi_swap(p, q);
        }
        gcry_mpi_invm(u, p, q);
    }

    if (compute_mul_inv) {
        err = gcry_sexp_build(&rsa->key, NULL,
            "(private-key (rsa (n %m) (e %m) (d %m) (p %m) (q %m) (u %m)))",
            n, e, d, p, q, u);
    } else {
        err = gcry_sexp_build(&rsa->key, NULL,
            "(private-key (rsa (n %m) (e %m) (d %m)))", n, e, d);
    }
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to build RSA private key: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }
    qcrypto_gcrypt_set_rsa_size((QCryptoAkCipher *)rsa,  n);
    ret = 0;

cleanup:
    gcry_mpi_release(n);
    gcry_mpi_release(e);
    gcry_mpi_release(d);
    gcry_mpi_release(p);
    gcry_mpi_release(q);
    gcry_mpi_release(u);
    return ret;
}

static int qcrypto_gcrypt_parse_rsa_public_key(QCryptoGcryptRSA *rsa,
                                               const uint8_t *key,
                                               size_t keylen,
                                               Error **errp)
{

    g_autoptr(QCryptoAkCipherRSAKey) rsa_key = qcrypto_akcipher_rsakey_parse(
        QCRYPTO_AK_CIPHER_KEY_TYPE_PUBLIC, key, keylen, errp);
    gcry_mpi_t n = NULL, e = NULL;
    int ret = -1;
    gcry_error_t err;

    if (!rsa_key) {
        return ret;
    }

    err = gcry_mpi_scan(&n, GCRYMPI_FMT_STD,
                        rsa_key->n.data, rsa_key->n.len, NULL);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to parse RSA parameter n: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    err = gcry_mpi_scan(&e, GCRYMPI_FMT_STD,
                        rsa_key->e.data, rsa_key->e.len, NULL);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to parse RSA parameter e: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    err = gcry_sexp_build(&rsa->key, NULL,
                          "(public-key (rsa (n %m) (e %m)))", n, e);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to build RSA public key: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }
    qcrypto_gcrypt_set_rsa_size((QCryptoAkCipher *)rsa, n);
    ret = 0;

cleanup:
    gcry_mpi_release(n);
    gcry_mpi_release(e);
    return ret;
}

static int qcrypto_gcrypt_rsa_encrypt(QCryptoAkCipher *akcipher,
                                      const void *in, size_t in_len,
                                      void *out, size_t out_len,
                                      Error **errp)
{
    QCryptoGcryptRSA *rsa = (QCryptoGcryptRSA *)akcipher;
    int ret = -1;
    gcry_sexp_t data_sexp = NULL, cipher_sexp = NULL;
    gcry_sexp_t cipher_sexp_item = NULL;
    gcry_mpi_t cipher_mpi = NULL;
    const char *result;
    gcry_error_t err;
    size_t actual_len;

    if (in_len > akcipher->max_plaintext_len) {
        error_setg(errp, "Plaintext length is greater than key size: %d",
                   akcipher->max_plaintext_len);
        return ret;
    }

    err = gcry_sexp_build(&data_sexp, NULL,
                          "(data (flags %s) (value %b))",
                          QCryptoRSAPaddingAlgo_str(rsa->padding_alg),
                          in_len, in);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to build plaintext: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    err = gcry_pk_encrypt(&cipher_sexp, data_sexp, rsa->key);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to encrypt: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    /* S-expression of cipher: (enc-val (rsa (a a-mpi))) */
    cipher_sexp_item = gcry_sexp_find_token(cipher_sexp, "a", 0);
    if (!cipher_sexp_item || gcry_sexp_length(cipher_sexp_item) != 2) {
        error_setg(errp, "Invalid ciphertext result");
        goto cleanup;
    }

    if (rsa->padding_alg == QCRYPTO_RSA_PADDING_ALGO_RAW) {
        cipher_mpi = gcry_sexp_nth_mpi(cipher_sexp_item, 1, GCRYMPI_FMT_USG);
        if (!cipher_mpi) {
            error_setg(errp, "Invalid ciphertext result");
            goto cleanup;
        }
        err = gcry_mpi_print(GCRYMPI_FMT_USG, out, out_len,
                             &actual_len, cipher_mpi);
        if (gcry_err_code(err) != 0) {
            error_setg(errp, "Failed to print MPI: %s/%s",
                       gcry_strsource(err), gcry_strerror(err));
            goto cleanup;
        }

        if (actual_len > out_len) {
            error_setg(errp, "Ciphertext buffer length is too small");
            goto cleanup;
        }

        /* We always padding leading-zeros for RSA-RAW */
        if (actual_len < out_len) {
            memmove((uint8_t *)out + (out_len - actual_len), out, actual_len);
            memset(out, 0, out_len - actual_len);
        }
        ret = out_len;

    } else {
        result = gcry_sexp_nth_data(cipher_sexp_item, 1, &actual_len);
        if (!result) {
            error_setg(errp, "Invalid ciphertext result");
            goto cleanup;
        }
        if (actual_len > out_len) {
            error_setg(errp, "Ciphertext buffer length is too small");
            goto cleanup;
        }
        memcpy(out, result, actual_len);
        ret = actual_len;
    }

cleanup:
    gcry_sexp_release(data_sexp);
    gcry_sexp_release(cipher_sexp);
    gcry_sexp_release(cipher_sexp_item);
    gcry_mpi_release(cipher_mpi);
    return ret;
}

static int qcrypto_gcrypt_rsa_decrypt(QCryptoAkCipher *akcipher,
                                      const void *in, size_t in_len,
                                      void *out, size_t out_len,
                                      Error **errp)
{
    QCryptoGcryptRSA *rsa = (QCryptoGcryptRSA *)akcipher;
    int ret = -1;
    gcry_sexp_t data_sexp = NULL, cipher_sexp = NULL;
    gcry_mpi_t data_mpi = NULL;
    gcry_error_t err;
    size_t actual_len;
    const char *result;

    if (in_len > akcipher->max_ciphertext_len) {
        error_setg(errp, "Ciphertext length is greater than key size: %d",
                   akcipher->max_ciphertext_len);
        return ret;
    }

    err = gcry_sexp_build(&cipher_sexp, NULL,
                          "(enc-val (flags %s) (rsa (a %b) ))",
                          QCryptoRSAPaddingAlgo_str(rsa->padding_alg),
                          in_len, in);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to build ciphertext: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    err = gcry_pk_decrypt(&data_sexp, cipher_sexp, rsa->key);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to decrypt: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    /* S-expression of plaintext: (value plaintext) */
    if (rsa->padding_alg == QCRYPTO_RSA_PADDING_ALGO_RAW) {
        data_mpi = gcry_sexp_nth_mpi(data_sexp, 1, GCRYMPI_FMT_USG);
        if (!data_mpi) {
            error_setg(errp, "Invalid plaintext result");
            goto cleanup;
        }
        err = gcry_mpi_print(GCRYMPI_FMT_USG, out, out_len,
                             &actual_len, data_mpi);
        if (gcry_err_code(err) != 0) {
            error_setg(errp, "Failed to print MPI: %s/%s",
                       gcry_strsource(err), gcry_strerror(err));
            goto cleanup;
        }
        if (actual_len > out_len) {
            error_setg(errp, "Plaintext buffer length is too small");
            goto cleanup;
        }
        /* We always padding leading-zeros for RSA-RAW */
        if (actual_len < out_len) {
            memmove((uint8_t *)out + (out_len - actual_len), out, actual_len);
            memset(out, 0, out_len - actual_len);
        }
        ret = out_len;
    } else {
        result = gcry_sexp_nth_data(data_sexp, 1, &actual_len);
        if (!result) {
            error_setg(errp, "Invalid plaintext result");
            goto cleanup;
        }
        if (actual_len > out_len) {
            error_setg(errp, "Plaintext buffer length is too small");
            goto cleanup;
        }
        memcpy(out, result, actual_len);
        ret = actual_len;
    }

cleanup:
    gcry_sexp_release(cipher_sexp);
    gcry_sexp_release(data_sexp);
    gcry_mpi_release(data_mpi);
    return ret;
}

static int qcrypto_gcrypt_rsa_sign(QCryptoAkCipher *akcipher,
                                   const void *in, size_t in_len,
                                   void *out, size_t out_len, Error **errp)
{
    QCryptoGcryptRSA *rsa = (QCryptoGcryptRSA *)akcipher;
    int ret = -1;
    gcry_sexp_t dgst_sexp = NULL, sig_sexp = NULL;
    gcry_sexp_t sig_sexp_item = NULL;
    const char *result;
    gcry_error_t err;
    size_t actual_len;

    if (in_len > akcipher->max_dgst_len) {
        error_setg(errp, "Data length is greater than key size: %d",
                   akcipher->max_dgst_len);
        return ret;
    }

    if (rsa->padding_alg != QCRYPTO_RSA_PADDING_ALGO_PKCS1) {
        error_setg(errp, "Invalid padding %u", rsa->padding_alg);
        return ret;
    }

    err = gcry_sexp_build(&dgst_sexp, NULL,
                          "(data (flags pkcs1) (hash %s %b))",
                          QCryptoHashAlgo_str(rsa->hash_alg),
                          in_len, in);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to build dgst: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    err = gcry_pk_sign(&sig_sexp, dgst_sexp, rsa->key);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to make signature: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    /* S-expression of signature: (sig-val (rsa (s s-mpi))) */
    sig_sexp_item = gcry_sexp_find_token(sig_sexp, "s", 0);
    if (!sig_sexp_item || gcry_sexp_length(sig_sexp_item) != 2) {
        error_setg(errp, "Invalid signature result");
        goto cleanup;
    }

    result = gcry_sexp_nth_data(sig_sexp_item, 1, &actual_len);
    if (!result) {
        error_setg(errp, "Invalid signature result");
        goto cleanup;
    }

    if (actual_len > out_len) {
        error_setg(errp, "Signature buffer length is too small");
        goto cleanup;
    }
    memcpy(out, result, actual_len);
    ret = actual_len;

cleanup:
    gcry_sexp_release(dgst_sexp);
    gcry_sexp_release(sig_sexp);
    gcry_sexp_release(sig_sexp_item);

    return ret;
}

static int qcrypto_gcrypt_rsa_verify(QCryptoAkCipher *akcipher,
                                     const void *in, size_t in_len,
                                     const void *in2, size_t in2_len,
                                     Error **errp)
{
    QCryptoGcryptRSA *rsa = (QCryptoGcryptRSA *)akcipher;
    int ret = -1;
    gcry_sexp_t sig_sexp = NULL, dgst_sexp = NULL;
    gcry_error_t err;

    if (in_len > akcipher->max_signature_len) {
        error_setg(errp, "Signature length is greater than key size: %d",
                   akcipher->max_signature_len);
        return ret;
    }

    if (in2_len > akcipher->max_dgst_len) {
        error_setg(errp, "Data length is greater than key size: %d",
                   akcipher->max_dgst_len);
        return ret;
    }

    if (rsa->padding_alg != QCRYPTO_RSA_PADDING_ALGO_PKCS1) {
        error_setg(errp, "Invalid padding %u", rsa->padding_alg);
        return ret;
    }

    err = gcry_sexp_build(&sig_sexp, NULL,
                          "(sig-val (rsa (s %b)))", in_len, in);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to build signature: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    err = gcry_sexp_build(&dgst_sexp, NULL,
                          "(data (flags pkcs1) (hash %s %b))",
                          QCryptoHashAlgo_str(rsa->hash_alg),
                          in2_len, in2);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to build dgst: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    err = gcry_pk_verify(sig_sexp, dgst_sexp, rsa->key);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to verify signature: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }
    ret = 0;

cleanup:
    gcry_sexp_release(dgst_sexp);
    gcry_sexp_release(sig_sexp);

    return ret;
}

QCryptoAkCipherDriver gcrypt_rsa = {
    .encrypt = qcrypto_gcrypt_rsa_encrypt,
    .decrypt = qcrypto_gcrypt_rsa_decrypt,
    .sign = qcrypto_gcrypt_rsa_sign,
    .verify = qcrypto_gcrypt_rsa_verify,
    .free = qcrypto_gcrypt_rsa_free,
};

static QCryptoGcryptRSA *qcrypto_gcrypt_rsa_new(
    const QCryptoAkCipherOptionsRSA *opt,
    QCryptoAkCipherKeyType type,
    const uint8_t *key, size_t keylen,
    Error **errp)
{
    QCryptoGcryptRSA *rsa = g_new0(QCryptoGcryptRSA, 1);
    rsa->padding_alg = opt->padding_alg;
    rsa->hash_alg = opt->hash_alg;
    rsa->akcipher.driver = &gcrypt_rsa;

    switch (type) {
    case QCRYPTO_AK_CIPHER_KEY_TYPE_PRIVATE:
        if (qcrypto_gcrypt_parse_rsa_private_key(rsa, key, keylen, errp) != 0) {
            goto error;
        }
        break;

    case QCRYPTO_AK_CIPHER_KEY_TYPE_PUBLIC:
        if (qcrypto_gcrypt_parse_rsa_public_key(rsa, key, keylen, errp) != 0) {
            goto error;
        }
        break;

    default:
        error_setg(errp, "Unknown akcipher key type %d", type);
        goto error;
    }

    return rsa;

error:
    qcrypto_gcrypt_rsa_free((QCryptoAkCipher *)rsa);
    return NULL;
}


/*
 * ECDSA support (sign and verify)
 *
 * Keys and signatures use raw big-endian formats:
 *   - public key: Qx || Qy, each 'coord_len' bytes
 *   - private key: the scalar d, 'coord_len' bytes
 *   - signature: r || s, each 'coord_len' bytes
 *   - the input to sign/verify is a raw message digest
 */
static const char *qcrypto_gcrypt_ecdsa_curve_name(QCryptoCurveID curve_id)
{
    switch (curve_id) {
    case QCRYPTO_CURVE_ID_PRIME256V1:
        return "NIST P-256";

    case QCRYPTO_CURVE_ID_SECP384R1:
        return "NIST P-384";

    default:
        return NULL;
    }
}

static size_t qcrypto_gcrypt_ecdsa_coord_len(QCryptoCurveID curve_id)
{
    switch (curve_id) {
    case QCRYPTO_CURVE_ID_PRIME256V1:
        return 32;

    case QCRYPTO_CURVE_ID_SECP384R1:
        return 48;

    default:
        return 0;
    }
}

static void qcrypto_gcrypt_ecdsa_free(QCryptoAkCipher *akcipher)
{
    QCryptoGcryptECDSA *ecdsa = (QCryptoGcryptECDSA *)akcipher;
    if (!ecdsa) {
        return;
    }

    gcry_sexp_release(ecdsa->key);
    g_free(ecdsa);
}

static int qcrypto_gcrypt_ecdsa_encrypt(QCryptoAkCipher *akcipher,
                                        const void *in, size_t in_len,
                                        void *out, size_t out_len,
                                        Error **errp)
{
    error_setg(errp, "ECDSA does not support encryption");
    return -1;
}

static int qcrypto_gcrypt_ecdsa_decrypt(QCryptoAkCipher *akcipher,
                                        const void *in, size_t in_len,
                                        void *out, size_t out_len,
                                        Error **errp)
{
    error_setg(errp, "ECDSA does not support decryption");
    return -1;
}

/*
 * Write an MPI into a fixed-length, big-endian, left-zero-padded buffer.
 *
 * gcry_mpi_print(GCRYMPI_FMT_USG) emits only the minimal number of bytes (it
 * drops leading zeros), but an ECDSA r/s component must occupy exactly the
 * curve size. Zero-fill the leading bytes and right-align the value, so a
 * component whose most significant byte is zero still lands at the correct
 * offset in the r || s output.
 *
 * Returns -1 if the value does not fit in 'len' bytes.
 */
static int qcrypto_gcrypt_mpi_to_buf(gcry_mpi_t mpi, uint8_t *buf, size_t len)
{
    size_t nbytes = (gcry_mpi_get_nbits(mpi) + 7) / 8;

    if (nbytes > len) {
        return -1;
    }
    memset(buf, 0, len - nbytes);
    gcry_mpi_print(GCRYMPI_FMT_USG, buf + (len - nbytes), nbytes, NULL, mpi);
    return 0;
}

static int qcrypto_gcrypt_ecdsa_sign(QCryptoAkCipher *akcipher,
                                     const void *in, size_t in_len,
                                     void *out, size_t out_len,
                                     Error **errp)
{
    QCryptoGcryptECDSA *ecdsa = (QCryptoGcryptECDSA *)akcipher;
    size_t coord_len = qcrypto_gcrypt_ecdsa_coord_len(ecdsa->curve_id);
    gcry_sexp_t dgst_sexp = NULL;
    gcry_sexp_t sig_sexp = NULL;
    gcry_sexp_t r_sexp = NULL;
    gcry_sexp_t s_sexp = NULL;
    gcry_mpi_t r_mpi = NULL;
    gcry_mpi_t s_mpi = NULL;
    gcry_error_t err;
    int ret = -1;

    if (ecdsa->type != QCRYPTO_AK_CIPHER_KEY_TYPE_PRIVATE) {
        error_setg(errp, "ECDSA sign requires a private key");
        return ret;
    }

    if (in_len == 0 || in_len > akcipher->max_dgst_len) {
        error_setg(errp, "Invalid digest length %zu", in_len);
        return ret;
    }

    if (out_len < coord_len * 2) {
        error_setg(errp, "Signature buffer length %zu is less than %zu",
                   out_len, coord_len * 2);
        return ret;
    }

    err = gcry_sexp_build(&dgst_sexp, NULL,
                          "(data (flags raw) (value %b))",
                          (int)in_len, in);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to build digest: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    err = gcry_pk_sign(&sig_sexp, dgst_sexp, ecdsa->key);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to make signature: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    /* S-expression of signature: (sig-val (ecdsa (r r-mpi) (s s-mpi))) */
    r_sexp = gcry_sexp_find_token(sig_sexp, "r", 0);
    s_sexp = gcry_sexp_find_token(sig_sexp, "s", 0);
    if (!r_sexp || !s_sexp) {
        error_setg(errp, "Invalid signature result");
        goto cleanup;
    }
    r_mpi = gcry_sexp_nth_mpi(r_sexp, 1, GCRYMPI_FMT_USG);
    s_mpi = gcry_sexp_nth_mpi(s_sexp, 1, GCRYMPI_FMT_USG);
    if (!r_mpi || !s_mpi) {
        error_setg(errp, "Invalid signature result");
        goto cleanup;
    }

    /* output is r || s, each zero-padded to the curve size */
    if (qcrypto_gcrypt_mpi_to_buf(r_mpi, out, coord_len) < 0 ||
        qcrypto_gcrypt_mpi_to_buf(s_mpi, (uint8_t *)out + coord_len,
                                  coord_len) < 0) {
        error_setg(errp, "Signature component is too large");
        goto cleanup;
    }
    ret = coord_len * 2;

cleanup:
    gcry_sexp_release(dgst_sexp);
    gcry_sexp_release(sig_sexp);
    gcry_sexp_release(r_sexp);
    gcry_sexp_release(s_sexp);
    gcry_mpi_release(r_mpi);
    gcry_mpi_release(s_mpi);
    return ret;
}

static int qcrypto_gcrypt_ecdsa_verify(QCryptoAkCipher *akcipher,
                                       const void *in, size_t in_len,
                                       const void *in2, size_t in2_len,
                                       Error **errp)
{
    QCryptoGcryptECDSA *ecdsa = (QCryptoGcryptECDSA *)akcipher;
    size_t coord_len = qcrypto_gcrypt_ecdsa_coord_len(ecdsa->curve_id);
    gcry_sexp_t sig_sexp = NULL;
    gcry_sexp_t dgst_sexp = NULL;
    gcry_error_t err;
    int ret = -1;

    if (ecdsa->type != QCRYPTO_AK_CIPHER_KEY_TYPE_PUBLIC) {
        error_setg(errp, "ECDSA verify requires a public key");
        return ret;
    }

    /* signature is r || s */
    if (in_len != coord_len * 2) {
        error_setg(errp, "Signature length %zu is not %zu",
                   in_len, coord_len * 2);
        return ret;
    }

    if (in2_len == 0 || in2_len > akcipher->max_dgst_len) {
        error_setg(errp, "Invalid digest length %zu", in2_len);
        return ret;
    }

    err = gcry_sexp_build(&sig_sexp, NULL,
                          "(sig-val (ecdsa (r %b) (s %b)))",
                          (int)coord_len, in,
                          (int)coord_len, (const uint8_t *)in + coord_len);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to build signature: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    err = gcry_sexp_build(&dgst_sexp, NULL,
                          "(data (flags raw) (value %b))",
                          (int)in2_len, in2);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to build digest: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }

    err = gcry_pk_verify(sig_sexp, dgst_sexp, ecdsa->key);
    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to verify signature: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto cleanup;
    }
    ret = 0;

cleanup:
    gcry_sexp_release(sig_sexp);
    gcry_sexp_release(dgst_sexp);
    return ret;
}

QCryptoAkCipherDriver gcrypt_ecdsa = {
    .encrypt = qcrypto_gcrypt_ecdsa_encrypt,
    .decrypt = qcrypto_gcrypt_ecdsa_decrypt,
    .sign = qcrypto_gcrypt_ecdsa_sign,
    .verify = qcrypto_gcrypt_ecdsa_verify,
    .free = qcrypto_gcrypt_ecdsa_free,
};

static QCryptoGcryptECDSA *qcrypto_gcrypt_ecdsa_new(
    const QCryptoAkCipherOptionsECDSA *opt,
    QCryptoAkCipherKeyType type,
    const uint8_t *key, size_t keylen,
    Error **errp)
{
    QCryptoGcryptECDSA *ecdsa;
    const char *curve_name = qcrypto_gcrypt_ecdsa_curve_name(opt->curve_id);
    size_t coord_len = qcrypto_gcrypt_ecdsa_coord_len(opt->curve_id);
    g_autofree uint8_t *point = NULL;
    gcry_error_t err;

    if (!curve_name || coord_len == 0) {
        error_setg(errp, "Unsupported curve id: %u", opt->curve_id);
        return NULL;
    }

    ecdsa = g_new0(QCryptoGcryptECDSA, 1);
    ecdsa->akcipher.driver = &gcrypt_ecdsa;
    ecdsa->type = type;
    ecdsa->curve_id = opt->curve_id;
    ecdsa->akcipher.max_dgst_len = coord_len;
    ecdsa->akcipher.max_signature_len = coord_len * 2;

    switch (type) {
    case QCRYPTO_AK_CIPHER_KEY_TYPE_PUBLIC:
        /* public key: Qx || Qy */
        if (keylen != coord_len * 2) {
            error_setg(errp, "Public key length %zu is not %zu",
                       keylen, coord_len * 2);
            goto error;
        }
        /* build uncompressed EC point: 0x04 || Qx || Qy */
        point = g_malloc(1 + keylen);
        point[0] = 0x04;
        memcpy(point + 1, key, keylen);
        err = gcry_sexp_build(&ecdsa->key, NULL,
                              "(public-key (ecc (curve %s) (q %b)))",
                              curve_name, (int)(1 + keylen), point);
        break;

    case QCRYPTO_AK_CIPHER_KEY_TYPE_PRIVATE:
        /* private key: the scalar d */
        if (keylen != coord_len) {
            error_setg(errp, "Private key length %zu is not %zu",
                       keylen, coord_len);
            goto error;
        }
        err = gcry_sexp_build(&ecdsa->key, NULL,
                              "(private-key (ecc (curve %s) (d %b)))",
                              curve_name, (int)keylen, key);
        break;

    default:
        error_setg(errp, "Unknown akcipher key type %d", type);
        goto error;
    }

    if (gcry_err_code(err) != 0) {
        error_setg(errp, "Failed to build ECDSA key: %s/%s",
                   gcry_strsource(err), gcry_strerror(err));
        goto error;
    }

    return ecdsa;

error:
    qcrypto_gcrypt_ecdsa_free((QCryptoAkCipher *)ecdsa);
    return NULL;
}

bool qcrypto_akcipher_supports(QCryptoAkCipherOptions *opts)
{
    switch (opts->alg) {
    case QCRYPTO_AK_CIPHER_ALGO_RSA:
        switch (opts->u.rsa.padding_alg) {
        case QCRYPTO_RSA_PADDING_ALGO_RAW:
            return true;

        case QCRYPTO_RSA_PADDING_ALGO_PKCS1:
            switch (opts->u.rsa.hash_alg) {
            case QCRYPTO_HASH_ALGO_MD5:
            case QCRYPTO_HASH_ALGO_SHA1:
            case QCRYPTO_HASH_ALGO_SHA256:
            case QCRYPTO_HASH_ALGO_SHA512:
                return true;

            default:
                return false;
            }

        default:
            return false;
        }

    case QCRYPTO_AK_CIPHER_ALGO_ECDSA:
        switch (opts->u.ecdsa.curve_id) {
        case QCRYPTO_CURVE_ID_PRIME256V1:
        case QCRYPTO_CURVE_ID_SECP384R1:
            return true;

        default:
            return false;
        }

    default:
        return false;
    }
}
