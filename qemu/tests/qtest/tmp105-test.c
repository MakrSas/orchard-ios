/*
 * QTest testcase for the TMP105 temperature sensor
 *
 * Copyright (c) 2012 Andreas Färber
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"

#include "libqtest-single.h"
#include "libqos/qgraph.h"
#include "libqos/i2c.h"
#include "qobject/qdict.h"
#include "hw/sensor/tmp105_regs.h"

#define TMP105_TEST_ID   "tmp105-test"
#define TMP105_TEST_ADDR 0x49
#define TMP105_TEST_PATH "/machine/peripheral/" TMP105_TEST_ID

#define TMP75_TEST_ID    "tmp75-test"
#define TMP75_TEST_PATH  "/machine/peripheral/" TMP75_TEST_ID

#define TMP175_TEST_ID   "tmp175-test"
#define TMP175_TEST_PATH "/machine/peripheral/" TMP175_TEST_ID

#define LM75B_TEST_ID    "lm75b-test"
#define LM75B_TEST_PATH  "/machine/peripheral/" LM75B_TEST_ID

#define TMP105_CONFIG_POL   (1 << 2)   /* ALERT active-high when set */
#define TMP105_CONFIG_TM    (1 << 1)   /* interrupt (thermostat) mode */
#define TMP105_CONFIG_FQ_1  (0 << 3)   /* fault queue: 1 consecutive fault */
#define TMP105_CONFIG_FQ_4  (2 << 3)   /* fault queue: 4 consecutive faults */
#define TMP105_CONFIG_FQ(f) ((f) << 3) /* raw F1:F0 fault-queue field value */
#define TMP105_CONFIG_SD    (1 << 0)   /* shutdown mode */
#define TMP105_CONFIG_OS    (1 << 7)   /* one-shot conversion */

static int qmp_tmp105_get_temperature(const char *id)
{
    QDict *response;
    int ret;

    response = qmp("{ 'execute': 'qom-get', 'arguments': { 'path': %s, "
                   "'property': 'temperature' } }", id);
    g_assert(qdict_haskey(response, "return"));
    ret = qdict_get_int(response, "return");
    qobject_unref(response);
    return ret;
}

static void qmp_tmp105_set_temperature(const char *id, int value)
{
    QDict *response;

    response = qmp("{ 'execute': 'qom-set', 'arguments': { 'path': %s, "
                   "'property': 'temperature', 'value': %d } }", id, value);
    g_assert(qdict_haskey(response, "return"));
    qobject_unref(response);
}

#define TMP105_PRECISION (1000/16)
static void send_and_receive(void *obj, void *data, QGuestAllocator *alloc)
{
    uint16_t value;
    QI2CDevice *i2cdev = (QI2CDevice *)obj;

    value = qmp_tmp105_get_temperature(TMP105_TEST_ID);
    g_assert_cmpuint(value, ==, 0);

    value = i2c_get16(i2cdev, TMP105_REG_TEMPERATURE);
    g_assert_cmphex(value, ==, 0);

    qmp_tmp105_set_temperature(TMP105_TEST_ID, 20000);
    value = qmp_tmp105_get_temperature(TMP105_TEST_ID);
    g_assert_cmpuint(value, ==, 20000);

    value = i2c_get16(i2cdev, TMP105_REG_TEMPERATURE);
    g_assert_cmphex(value, ==, 0x1400);

    qmp_tmp105_set_temperature(TMP105_TEST_ID, 20938); /* 20 + 15/16 */
    value = qmp_tmp105_get_temperature(TMP105_TEST_ID);
    g_assert_cmpuint(value, >=, 20938 - TMP105_PRECISION/2);
    g_assert_cmpuint(value, <, 20938 + TMP105_PRECISION/2);

    /* Set config */
    i2c_set8(i2cdev, TMP105_REG_CONFIG, 0x60);
    value = i2c_get8(i2cdev, TMP105_REG_CONFIG);
    g_assert_cmphex(value, ==, 0x60);

    value = i2c_get16(i2cdev, TMP105_REG_TEMPERATURE);
    g_assert_cmphex(value, ==, 0x14f0);

    /* Set precision to 9, 10, 11 bits.  */
    i2c_set8(i2cdev, TMP105_REG_CONFIG, 0x00);
    g_assert_cmphex(i2c_get8(i2cdev, TMP105_REG_CONFIG), ==, 0x00);
    value = i2c_get16(i2cdev, TMP105_REG_TEMPERATURE);
    g_assert_cmphex(value, ==, 0x1480);

    i2c_set8(i2cdev, TMP105_REG_CONFIG, 0x20);
    g_assert_cmphex(i2c_get8(i2cdev, TMP105_REG_CONFIG), ==, 0x20);
    value = i2c_get16(i2cdev, TMP105_REG_TEMPERATURE);
    g_assert_cmphex(value, ==, 0x14c0);

    i2c_set8(i2cdev, TMP105_REG_CONFIG, 0x40);
    g_assert_cmphex(i2c_get8(i2cdev, TMP105_REG_CONFIG), ==, 0x40);
    value = i2c_get16(i2cdev, TMP105_REG_TEMPERATURE);
    g_assert_cmphex(value, ==, 0x14e0);

    /* stored precision remains the same */
    value = qmp_tmp105_get_temperature(TMP105_TEST_ID);
    g_assert_cmpuint(value, >=, 20938 - TMP105_PRECISION/2);
    g_assert_cmpuint(value, <, 20938 + TMP105_PRECISION/2);

    i2c_set8(i2cdev, TMP105_REG_CONFIG, 0x60);
    g_assert_cmphex(i2c_get8(i2cdev, TMP105_REG_CONFIG), ==, 0x60);
    value = i2c_get16(i2cdev, TMP105_REG_TEMPERATURE);
    g_assert_cmphex(value, ==, 0x14f0);

    i2c_set16(i2cdev, TMP105_REG_T_LOW, 0x1234);
    g_assert_cmphex(i2c_get16(i2cdev, TMP105_REG_T_LOW), ==, 0x1230);
    i2c_set16(i2cdev, TMP105_REG_T_HIGH, 0x4231);
    g_assert_cmphex(i2c_get16(i2cdev, TMP105_REG_T_HIGH), ==, 0x4230);
}

/*
 * The TMP105 exposes its alarm state only through the ALERT pin.
 */
static void test_alert_single_fault(void *obj, void *data,
                                    QGuestAllocator *alloc)
{
    QI2CDevice *i2cdev = (QI2CDevice *)obj;

    qtest_irq_intercept_out(global_qtest, TMP105_TEST_PATH);

    i2c_set8(i2cdev, TMP105_REG_CONFIG, TMP105_CONFIG_POL | TMP105_CONFIG_FQ_1);
    g_assert_false(get_irq(0));

    qmp_tmp105_set_temperature(TMP105_TEST_ID, 85000);
    g_assert_true(get_irq(0));

    qmp_tmp105_set_temperature(TMP105_TEST_ID, 70000);
    g_assert_false(get_irq(0));
}

static void test_fault_queue(void *obj, void *data, QGuestAllocator *alloc)
{
    QI2CDevice *i2cdev = (QI2CDevice *)obj;
    int i;

    qtest_irq_intercept_out(global_qtest, TMP105_TEST_PATH);

    /* Comparator mode, active-high ALERT, fault queue of four. */
    i2c_set8(i2cdev, TMP105_REG_CONFIG, TMP105_CONFIG_POL | TMP105_CONFIG_FQ_4);
    g_assert_false(get_irq(0));

    for (i = 0; i < 3; i++) {
        qmp_tmp105_set_temperature(TMP105_TEST_ID, 85000);
        g_assert_false(get_irq(0));
    }

    qmp_tmp105_set_temperature(TMP105_TEST_ID, 25000);
    g_assert_false(get_irq(0));

    for (i = 0; i < 3; i++) {
        qmp_tmp105_set_temperature(TMP105_TEST_ID, 85000);
        g_assert_false(get_irq(0));
    }

    qmp_tmp105_set_temperature(TMP105_TEST_ID, 85000);
    g_assert_true(get_irq(0));

    for (i = 0; i < 3; i++) {
        qmp_tmp105_set_temperature(TMP105_TEST_ID, 70000);
        g_assert_true(get_irq(0));
    }

    qmp_tmp105_set_temperature(TMP105_TEST_ID, 70000);
    g_assert_false(get_irq(0));
}

/*
 * Drive @need consecutive over-limit conversions and check that the ALERT pin
 * only asserts on the last one. This exercises the fault-queue length.
 */
static void check_fault_queue(QI2CDevice *i2cdev, const char *id,
                              const char *path, uint8_t fq_field, int need)
{
    int i;

    qtest_irq_intercept_out(global_qtest, path);

    i2c_set8(i2cdev, TMP105_REG_CONFIG,
             TMP105_CONFIG_POL | TMP105_CONFIG_FQ(fq_field));
    g_assert_false(get_irq(0));

    for (i = 0; i < need - 1; i++) {
        qmp_tmp105_set_temperature(id, 85000);
        g_assert_false(get_irq(0));
    }
    qmp_tmp105_set_temperature(id, 85000);
    g_assert_true(get_irq(0));
}

/*
 * The one-shot (OS) bit starts a conversion only in shutdown mode. In
 * continuous mode it is ignored, so writing it must not advance the fault
 * queue; in shutdown each OS write performs one conversion that does.
 */
static void test_one_shot(void *obj, void *data, QGuestAllocator *alloc)
{
    QI2CDevice *i2cdev = (QI2CDevice *)obj;
    int i;

    qtest_irq_intercept_out(global_qtest, TMP105_TEST_PATH);

    i2c_set8(i2cdev, TMP105_REG_CONFIG, TMP105_CONFIG_POL | TMP105_CONFIG_FQ_4);
    qmp_tmp105_set_temperature(TMP105_TEST_ID, 85000);
    g_assert_false(get_irq(0));

    for (i = 0; i < 8; i++) {
        i2c_set8(i2cdev, TMP105_REG_CONFIG,
                 TMP105_CONFIG_POL | TMP105_CONFIG_FQ_4 | TMP105_CONFIG_OS);
        g_assert_false(get_irq(0));
    }

    i2c_set8(i2cdev, TMP105_REG_CONFIG,
             TMP105_CONFIG_POL | TMP105_CONFIG_FQ_4 | TMP105_CONFIG_SD);
    for (i = 0; i < 2; i++) {
        i2c_set8(i2cdev, TMP105_REG_CONFIG, TMP105_CONFIG_POL |
                 TMP105_CONFIG_FQ_4 | TMP105_CONFIG_SD | TMP105_CONFIG_OS);
        g_assert_false(get_irq(0));
    }
    i2c_set8(i2cdev, TMP105_REG_CONFIG, TMP105_CONFIG_POL |
             TMP105_CONFIG_FQ_4 | TMP105_CONFIG_SD | TMP105_CONFIG_OS);
    g_assert_true(get_irq(0));
}

/*
 * Configuration and limit-register writes are not conversions and must not
 * advance the fault queue.
 */
static void test_fault_queue_ignores_writes(void *obj, void *data,
                                            QGuestAllocator *alloc)
{
    QI2CDevice *i2cdev = (QI2CDevice *)obj;
    int i;

    qtest_irq_intercept_out(global_qtest, TMP105_TEST_PATH);

    i2c_set8(i2cdev, TMP105_REG_CONFIG, TMP105_CONFIG_POL | TMP105_CONFIG_FQ_4);
    g_assert_false(get_irq(0));

    qmp_tmp105_set_temperature(TMP105_TEST_ID, 85000);
    g_assert_false(get_irq(0));

    for (i = 0; i < 8; i++) {
        i2c_set8(i2cdev, TMP105_REG_CONFIG,
                 TMP105_CONFIG_POL | TMP105_CONFIG_FQ_4);
        i2c_set16(i2cdev, TMP105_REG_T_HIGH, 0x5000);
        i2c_set16(i2cdev, TMP105_REG_T_LOW, 0x4b00);
        g_assert_false(get_irq(0));
    }

    for (i = 0; i < 2; i++) {
        qmp_tmp105_set_temperature(TMP105_TEST_ID, 85000);
        g_assert_false(get_irq(0));
    }
    qmp_tmp105_set_temperature(TMP105_TEST_ID, 85000);
    g_assert_true(get_irq(0));
}

/*
 * Leaving shutdown (SD 1->0) resumes continuous conversion, which must
 * re-evaluate the current temperature against the limits.
 */
static void test_wake_from_shutdown(void *obj, void *data,
                                    QGuestAllocator *alloc)
{
    QI2CDevice *i2cdev = (QI2CDevice *)obj;

    qtest_irq_intercept_out(global_qtest, TMP105_TEST_PATH);

    i2c_set8(i2cdev, TMP105_REG_CONFIG, TMP105_CONFIG_POL | TMP105_CONFIG_FQ_1);
    g_assert_false(get_irq(0));

    qmp_tmp105_set_temperature(TMP105_TEST_ID, 85000);
    g_assert_true(get_irq(0));

    i2c_set8(i2cdev, TMP105_REG_CONFIG,
             TMP105_CONFIG_POL | TMP105_CONFIG_FQ_1 | TMP105_CONFIG_SD);
    g_assert_true(get_irq(0));

    qmp_tmp105_set_temperature(TMP105_TEST_ID, 70000);
    g_assert_true(get_irq(0));

    i2c_set8(i2cdev, TMP105_REG_CONFIG, TMP105_CONFIG_POL | TMP105_CONFIG_FQ_1);
    g_assert_false(get_irq(0));
}

/* The TMP75 maps F1:F0 = 10b to 3 consecutive faults. */
static void test_tmp75_fault_queue(void *obj, void *data,
                                   QGuestAllocator *alloc)
{
    check_fault_queue(obj, TMP75_TEST_ID, TMP75_TEST_PATH, 2, 3);
}

/* The TMP175 keeps the TMP105 mapping: F1:F0 = 10b means 4 faults. */
static void test_tmp175_fault_queue(void *obj, void *data,
                                    QGuestAllocator *alloc)
{
    check_fault_queue(obj, TMP175_TEST_ID, TMP175_TEST_PATH, 2, 4);
}

/*
 * Toggling the thermostat mode (TM) bit clears any active alert on the TMP75.
 */
static void test_tmp75_tm_clears_alert(void *obj, void *data,
                                       QGuestAllocator *alloc)
{
    QI2CDevice *i2cdev = (QI2CDevice *)obj;
    int i;

    qtest_irq_intercept_out(global_qtest, TMP75_TEST_PATH);

    i2c_set8(i2cdev, TMP105_REG_CONFIG,
             TMP105_CONFIG_POL | TMP105_CONFIG_FQ(3));
    for (i = 0; i < 4; i++) {
        qmp_tmp105_set_temperature(TMP75_TEST_ID, 85000);
    }
    g_assert_true(get_irq(0));

    i2c_set8(i2cdev, TMP105_REG_CONFIG,
             TMP105_CONFIG_POL | TMP105_CONFIG_FQ(3) | TMP105_CONFIG_TM);
    g_assert_false(get_irq(0));
}

/*
 * Entering shutdown clears the ALERT in interrupt mode but leaves it asserted
 * in comparator mode.
 */
static void test_tmp75_shutdown_clears_alert(void *obj, void *data,
                                             QGuestAllocator *alloc)
{
    QI2CDevice *i2cdev = (QI2CDevice *)obj;
    int i;

    qtest_irq_intercept_out(global_qtest, TMP75_TEST_PATH);

    i2c_set8(i2cdev, TMP105_REG_CONFIG,
             TMP105_CONFIG_POL | TMP105_CONFIG_TM | TMP105_CONFIG_FQ(3));
    for (i = 0; i < 4; i++) {
        qmp_tmp105_set_temperature(TMP75_TEST_ID, 85000);
    }
    g_assert_true(get_irq(0));

    i2c_set8(i2cdev, TMP105_REG_CONFIG,
             TMP105_CONFIG_POL | TMP105_CONFIG_TM | TMP105_CONFIG_FQ(3) |
             TMP105_CONFIG_SD);
    g_assert_false(get_irq(0));

    i2c_set8(i2cdev, TMP105_REG_CONFIG,
             TMP105_CONFIG_POL | TMP105_CONFIG_FQ(3));
    for (i = 0; i < 4; i++) {
        qmp_tmp105_set_temperature(TMP75_TEST_ID, 85000);
    }
    g_assert_true(get_irq(0));
    i2c_set8(i2cdev, TMP105_REG_CONFIG,
             TMP105_CONFIG_POL | TMP105_CONFIG_FQ(3) | TMP105_CONFIG_SD);
    g_assert_true(get_irq(0));
}

/*
 * The LM75B has a fixed 11-bit (0.125 C) converter: the resolution and one-shot
 * Config bits are reserved (read/write as zero) and the temperature register is
 * always masked to 11 bits regardless of what is written to Config.
 */
static void test_lm75b_resolution(void *obj, void *data,
                                  QGuestAllocator *alloc)
{
    QI2CDevice *i2cdev = (QI2CDevice *)obj;
    uint16_t value;

    i2c_set8(i2cdev, TMP105_REG_CONFIG, 0x60);
    g_assert_cmphex(i2c_get8(i2cdev, TMP105_REG_CONFIG), ==, 0x00);

    qmp_tmp105_set_temperature(LM75B_TEST_ID, 20938);
    value = i2c_get16(i2cdev, TMP105_REG_TEMPERATURE);
    g_assert_cmphex(value, ==, 0x14e0);
}

/* The LM75B set-point registers store only 9 bits. */
static void test_lm75b_limits(void *obj, void *data,
                              QGuestAllocator *alloc)
{
    QI2CDevice *i2cdev = (QI2CDevice *)obj;

    i2c_set16(i2cdev, TMP105_REG_T_HIGH, 0x4231);
    g_assert_cmphex(i2c_get16(i2cdev, TMP105_REG_T_HIGH), ==, 0x4200);

    i2c_set16(i2cdev, TMP105_REG_T_LOW, 0x12b4);
    g_assert_cmphex(i2c_get16(i2cdev, TMP105_REG_T_LOW), ==, 0x1280);
}

/* The LM75B likewise resets its OS output on shutdown in interrupt mode. */
static void test_lm75b_shutdown_clears_alert(void *obj, void *data,
                                             QGuestAllocator *alloc)
{
    QI2CDevice *i2cdev = (QI2CDevice *)obj;

    qtest_irq_intercept_out(global_qtest, LM75B_TEST_PATH);

    i2c_set8(i2cdev, TMP105_REG_CONFIG, TMP105_CONFIG_POL | TMP105_CONFIG_TM);
    qmp_tmp105_set_temperature(LM75B_TEST_ID, 85000);
    g_assert_true(get_irq(0));

    i2c_set8(i2cdev, TMP105_REG_CONFIG,
             TMP105_CONFIG_POL | TMP105_CONFIG_TM | TMP105_CONFIG_SD);
    g_assert_false(get_irq(0));
}

static void tmp105_register_nodes(void)
{
    QOSGraphEdgeOptions opts = {
        .extra_device_opts = "id=" TMP105_TEST_ID ",address=0x49"
    };
    add_qi2c_address(&opts, &(QI2CAddress) { 0x49 });

    qos_node_create_driver("tmp105", i2c_device_create);
    qos_node_consumes("tmp105", "i2c-bus", &opts);

    qos_add_test("tx-rx", "tmp105", send_and_receive, NULL);
    qos_add_test("alert-single-fault", "tmp105", test_alert_single_fault, NULL);
    qos_add_test("fault-queue", "tmp105", test_fault_queue, NULL);
    qos_add_test("fault-queue-ignores-writes", "tmp105",
                 test_fault_queue_ignores_writes, NULL);
    qos_add_test("one-shot", "tmp105", test_one_shot, NULL);
    qos_add_test("wake-from-shutdown", "tmp105", test_wake_from_shutdown, NULL);

    /* TMP75: register-compatible, but with a 1/2/3/4 fault queue. */
    QOSGraphEdgeOptions tmp75_opts = {
        .extra_device_opts = "id=" TMP75_TEST_ID ",address=0x48"
    };
    add_qi2c_address(&tmp75_opts, &(QI2CAddress) { 0x48 });

    qos_node_create_driver("tmp75", i2c_device_create);
    qos_node_consumes("tmp75", "i2c-bus", &tmp75_opts);

    qos_add_test("fault-queue", "tmp75", test_tmp75_fault_queue, NULL);
    qos_add_test("tm-clears-alert", "tmp75", test_tmp75_tm_clears_alert, NULL);
    qos_add_test("shutdown-clears-alert", "tmp75",
                 test_tmp75_shutdown_clears_alert, NULL);

    /* TMP175: like the TMP105, with a 1/2/4/6 fault queue. */
    QOSGraphEdgeOptions tmp175_opts = {
        .extra_device_opts = "id=" TMP175_TEST_ID ",address=0x4a"
    };
    add_qi2c_address(&tmp175_opts, &(QI2CAddress) { 0x4a });

    qos_node_create_driver("tmp175", i2c_device_create);
    qos_node_consumes("tmp175", "i2c-bus", &tmp175_opts);

    qos_add_test("fault-queue", "tmp175", test_tmp175_fault_queue, NULL);

    /* LM75B: fixed 11-bit conversion and 9-bit set-point registers. */
    QOSGraphEdgeOptions lm75b_opts = {
        .extra_device_opts = "id=" LM75B_TEST_ID ",address=0x4c"
    };
    add_qi2c_address(&lm75b_opts, &(QI2CAddress) { 0x4c });

    qos_node_create_driver("lm75b", i2c_device_create);
    qos_node_consumes("lm75b", "i2c-bus", &lm75b_opts);

    qos_add_test("resolution", "lm75b", test_lm75b_resolution, NULL);
    qos_add_test("limits", "lm75b", test_lm75b_limits, NULL);
    qos_add_test("shutdown-clears-alert", "lm75b",
                 test_lm75b_shutdown_clears_alert, NULL);
}
libqos_init(tmp105_register_nodes);
