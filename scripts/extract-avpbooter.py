#!/usr/bin/env python3
"""Pull AVPBooter out of the macOS image, so the setup needs no Mac.

AVPBooter is Apple's VM firmware. It ships inside macOS, in
`Virtualization.framework` — and the macOS image this project already downloads
*is* macOS, so the firmware is right there in the guest's own system volume.

This mounts the image read-only with apfs-fuse, copies the firmware out, and
unmounts. It does not modify the image, and it writes nothing into it.

The copy inside the image is the build that matches it: the serial log names the
same `iBoot-8422.141.2.700.1` when the chain runs. Feed the result to
`patch-avpbooter.py`.

Copyright (c) 2026 Youssef Elliethy (yaelliethy)
SPDX-License-Identifier: GPL-2.0-or-later
"""

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile

FIRMWARE = ("System/Library/Frameworks/Virtualization.framework/"
            "Versions/A/Resources/AVPBooter.vmapple2.bin")


def system_container_offset(image):
    """Byte offset of the largest APFS container in the image's GPT.

    The tart layout is a 0.5 GiB `iBootSystemContainer` followed by the ~41 GiB
    `Container` that holds System and Data; the firmware is in the latter, which
    is also simply the biggest.
    """
    with open(image, "rb") as f:
        f.seek(512)
        header = f.read(92)
        if header[:8] != b"EFI PART":
            sys.exit(f"{image}: no GPT here")
        entry_lba, count, entry_size = struct.unpack_from("<QII", header, 72)
        f.seek(entry_lba * 512)
        biggest = None
        for _ in range(count):
            entry = f.read(entry_size)
            first, last = struct.unpack_from("<QQ", entry, 32)
            if not first:
                continue
            span = last - first
            if biggest is None or span > biggest[1]:
                biggest = (first * 512, span)
    if not biggest:
        sys.exit(f"{image}: no partitions in the GPT")
    return biggest[0]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default="images/disk.raw",
                    help="the macOS disk image (default: %(default)s)")
    ap.add_argument("-o", "--out", default="images/AVPBooter.vmapple2.bin",
                    help="where to write the firmware (default: %(default)s)")
    ap.add_argument("--apfs-fuse", default=os.environ.get("APFS_FUSE", "apfs-fuse"),
                    help="the apfs-fuse binary (default: %(default)s, or $APFS_FUSE)")
    args = ap.parse_args()

    if not shutil.which(args.apfs_fuse) and not os.path.exists(args.apfs_fuse):
        sys.exit("apfs-fuse not found: build it from "
                 "https://github.com/sgan81/apfs-fuse and pass --apfs-fuse")

    offset = system_container_offset(args.image)
    print(f"{args.image}: APFS container at {offset:#x}")

    mount = tempfile.mkdtemp(prefix="orchard-guest-")
    try:
        subprocess.run([args.apfs_fuse, "-o", "ro", "-s", str(offset),
                        args.image, mount], check=True)
        try:
            # apfs-fuse exposes the sealed system volume under `root/`.
            source = os.path.join(mount, "root", FIRMWARE)
            if not os.path.exists(source):
                sys.exit(f"not in this image: {FIRMWARE}")
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            shutil.copyfile(source, args.out)
        finally:
            subprocess.run(["fusermount3", "-u", mount],
                           check=False, stderr=subprocess.DEVNULL)
            subprocess.run(["fusermount", "-u", mount],
                           check=False, stderr=subprocess.DEVNULL)
    finally:
        os.rmdir(mount) if os.path.isdir(mount) and not os.listdir(mount) else None

    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes)")
    print(f"next: scripts/patch-avpbooter.py {args.out} images/AVPBooter.patched.bin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
