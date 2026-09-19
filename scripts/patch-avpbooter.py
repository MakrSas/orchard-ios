#!/usr/bin/env python3
"""Patch a host-supplied AVPBooter so it will run under QEMU.

AVPBooter is Apple's first-stage VM firmware. It ships inside macOS, in
`Virtualization.framework` — so the macOS image this project downloads already
carries it, and `extract-avpbooter.py` pulls it out. It is **not
redistributable**: extract your own, do not pass one around.

The patch itself is not ours: it comes from NyanSatan's Virtual-iBoot-Fun
(https://github.com/NyanSatan/Virtual-iBoot-Fun), which established what
AVPBooter checks and which routine has to be neutered to run it outside Apple's
own hypervisor. This script only applies it, and refuses to if the bytes at the
site are not the ones it expects.

What the patch does, and why it is only one:

  The stock ROM calls a routine early in its start-up that, on this emulated
  machine, never returns success; with it in place the boot stops before LLB is
  ever loaded. The patch replaces that routine's first two instructions with
  `mov x0, #0; ret`, i.e. makes it return 0 to its caller and touch nothing
  else. Everything after it — image4 verification, LLB, iBoot, XNU — runs
  stock.

The patch is applied by *pattern*, not by trusting an offset: the site must
begin with `pacibsp; sub sp, sp, #0x10` at the expected file offset. A ROM that
does not match is refused rather than corrupted.

Copyright (c) 2026 Youssef Elliethy (yaelliethy)
SPDX-License-Identifier: GPL-2.0-or-later
"""

import argparse
import hashlib
import shutil
import struct
import sys

# The site, as measured on AVPBooter.vmapple2.bin from macOS 13/14 hosts.
DEFAULT_OFFSET = 0x2314
EXPECTED = bytes.fromhex("7f2303d5" "ff4303d1")   # pacibsp ; sub sp, sp, #0x10
REPLACEMENT = bytes.fromhex("000080d2" "c0035fd6")  # mov x0, #0 ; ret


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rom", help="stock AVPBooter.vmapple2.bin")
    ap.add_argument("out", help="patched ROM to write")
    ap.add_argument("--offset", type=lambda v: int(v, 0), default=DEFAULT_OFFSET,
                    help="file offset of the routine (default: 0x2314)")
    args = ap.parse_args()

    data = bytearray(open(args.rom, "rb").read())
    print(f"input:  {args.rom}  {len(data)} bytes  "
          f"sha256 {hashlib.sha256(data).hexdigest()[:16]}…")

    here = bytes(data[args.offset:args.offset + len(EXPECTED)])
    if here == REPLACEMENT:
        print("already patched; nothing to do")
        shutil.copyfile(args.rom, args.out)
        return 0
    if here != EXPECTED:
        print(f"at {args.offset:#x} found {here.hex()}, expected {EXPECTED.hex()}",
              file=sys.stderr)
        print("refusing to patch a ROM that does not match; pass --offset if this "
              "is a firmware build whose routine sits elsewhere", file=sys.stderr)
        return 1

    data[args.offset:args.offset + len(REPLACEMENT)] = REPLACEMENT
    open(args.out, "wb").write(data)
    print(f"output: {args.out}  patched {len(REPLACEMENT)} bytes at {args.offset:#x}")
    print(f"        sha256 {hashlib.sha256(data).hexdigest()[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
