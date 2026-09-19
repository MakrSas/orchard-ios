/*
 * Texas Instruments TMP105/TMP75/TMP175/LM75B Temperature Sensor
 *
 * Browse the data sheet:
 *
 *    http://www.ti.com/lit/gpn/tmp105
 *
 * Copyright (C) 2012 Alex Horn <alex.horn@cs.ox.ac.uk>
 * Copyright (C) 2008-2012 Andrzej Zaborowski <balrogg@gmail.com>
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or
 * later. See the COPYING file in the top-level directory.
 */
#ifndef HW_SENSOR_TMP105_H
#define HW_SENSOR_TMP105_H

/* TMP75, TMP175 and NXP LM75B are register-compatible with TMP105. */
#define TYPE_TMP105 "tmp105"
#define TYPE_TMP175 "tmp175"
#define TYPE_TMP75  "tmp75"
#define TYPE_LM75B  "lm75b"

#endif
