#!/usr/bin/env python3
"""Append `boot-args` to a tart NVRAM (aux) image, in place of nothing else.

Why it is this careful:

The aux holds two CHRP NVRAM banks, at 0xa00000 (length 0x2000) and 0xa80000
(length 0x80000); the one with the higher generation is live. Each bank stores
an adler32 over `[base+0x14, base+banklen)` little-endian at `base+0x10`.

**Append-only works; re-serialising the entry list does not.** Rewriting the
list makes iBoot reject the NVRAM, set `iboot-failure-reason=0x65` and
`boot-command=recover`, and loop — measured at 523 iterations of iBootStage1
with no stage 2. So this appends one `KEY=VALUE\\0` after the last entry of the
live bank, fixes that bank's checksum, and changes nothing else: 25 bytes
differ for `boot-args=-v serial=3`.

Serial output is not otherwise reachable on this path — the AVPBooter chain has
no `-append`, because boot-args come from NVRAM.

Copyright (c) 2026 Youssef Elliethy (yaelliethy)
SPDX-License-Identifier: GPL-2.0-or-later
"""

import argparse
import shutil
import sys
import zlib

BANKS = ((0xA00000, 0x2000), (0xA80000, 0x80000))
GEN_OFF = 0x14          # generation counter, LE u32, first word of the payload
CKSUM_OFF = 0x10        # adler32 of [base+0x14, base+banklen), LE u32


def bank_generation(data, base):
    return int.from_bytes(data[base + GEN_OFF:base + GEN_OFF + 4], "little")


def live_bank(data):
    """The bank with the higher generation is the one iBoot reads."""
    best = None
    for base, length in BANKS:
        if base + length > len(data):
            continue
        gen = bank_generation(data, base)
        if best is None or gen > best[2]:
            best = (base, length, gen)
    return best


def last_entry_end(data, base, length):
    """Offset just past the final NUL-terminated entry in the bank."""
    start = base + GEN_OFF + 4
    end = base + length
    pos = start
    last = start
    while pos < end:
        nul = data.find(b"\x00", pos, end)
        if nul < 0 or nul == pos:      # empty entry: the list ends here
            break
        last = nul + 1
        pos = nul + 1
    return last


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aux", help="tart NVRAM image (modified in place unless --out)")
    ap.add_argument("--out", help="write here instead of patching in place")
    ap.add_argument("--set", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="entry to append, e.g. --set 'boot-args=-v serial=3'")
    ap.add_argument("--show", action="store_true",
                    help="print the live bank's entries and exit")
    args = ap.parse_args()

    data = bytearray(open(args.aux, "rb").read())
    found = live_bank(data)
    if not found:
        sys.exit("no CHRP bank found: is this a tart nvram image?")
    base, length, gen = found
    print(f"live bank at {base:#x}, length {length:#x}, generation {gen}")

    if args.show:
        end = last_entry_end(data, base, length)
        body = bytes(data[base + GEN_OFF + 4:end])
        for entry in body.split(b"\x00"):
            if entry:
                print("  " + entry.decode("utf-8", "replace"))
        return 0

    if not args.set:
        sys.exit("nothing to do: pass --set KEY=VALUE (or --show)")

    end = last_entry_end(data, base, length)
    for item in args.set:
        if "=" not in item:
            sys.exit(f"--set needs KEY=VALUE, got {item!r}")
        blob = item.encode() + b"\x00"
        if end + len(blob) > base + length:
            sys.exit("no room left in the bank for this entry")
        data[end:end + len(blob)] = blob
        end += len(blob)
        print(f"appended {item!r}")

    # Fix only this bank's checksum. The other bank is left untouched, and so
    # is the generation: iBoot re-reads the list, it does not diff it.
    checked = bytes(data[base + GEN_OFF:base + length])
    data[base + CKSUM_OFF:base + CKSUM_OFF + 4] = \
        (zlib.adler32(checked) & 0xFFFFFFFF).to_bytes(4, "little")

    out = args.out or args.aux
    if args.out:
        shutil.copyfile(args.aux, args.out)
    open(out, "wb").write(bytes(data))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
