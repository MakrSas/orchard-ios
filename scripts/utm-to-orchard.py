#!/usr/bin/env python3
"""Turn a UTM macOS virtual machine into the files this project boots from.

A UTM VM made with the "Virtualize" backend is a Virtualization.framework VM,
the same kind tart packages: a raw disk, an auxiliary storage file holding the
NVRAM and LocalPolicy, and a machine identifier carrying the ECID that policy
is signed for. This lays those out the way `scripts/run-vm.sh` and the iOS app
expect them, and adds the two things a VM never ships with:

  disk.qcow2             the system disk as qcow2 (default), or disk.raw with
                         --disk-format raw. qcow2 because the guest's disk is
                         64 GiB of which ~16 are used: a raw copy is only that
                         small while its holes survive, and exFAT, a copy
                         through the Files app or most transfers to a phone
                         fill them in. qcow2 simply does not store them. The
                         machine does not care: its "pflash" drives are block
                         backends for the boot device, not flash, and any
                         format QEMU reads will do.
                         An ASIF source — what UTM makes on macOS 26 and later,
                         magic `shdw`, which QEMU cannot read — is attached
                         read-only and read through its device node; a raw
                         source with --disk-format raw is cloned (APFS
                         clonefile: instant and free on the same volume).
  aux.img                the auxiliary storage without its 0x4000-byte header,
                         which is how QEMU's machine reads it (tart has the same
                         wrapper; see strip_nvram_header in fetch-tart-image.py)
  config.json            tart's format: `ecid` and `hardwareModel` are base64 of
                         the VM's own binary plists
  AVPBooter.patched.bin  Apple's VM firmware, taken out of the *guest's* own
                         Virtualization.framework and patched by
                         patch-avpbooter.py
  overlay.qcow2          a qcow2 over the disk for everything the guest writes,
                         with a relative backing path so the folder can move

The firmware comes from the guest, not the host, on purpose: patch-avpbooter.py
checks the bytes it replaces and knows the layout of the macOS 13/14 firmware.
A newer host's firmware has the routine elsewhere, and the patch rightly
refuses it rather than guess.

The VM must be shut down: a disk cloned from a running machine is a disk
caught halfway through writing.

Copyright (c) 2026 Orchard iOS port.
SPDX-License-Identifier: GPL-2.0-or-later
"""

import argparse
import base64
import json
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# Where the first CHRP NVRAM bank lives in the aux QEMU attaches, and the magic
# that identifies it. Same constants as fetch-tart-image.py, whose import would
# pull in `requests` for nothing.
NVRAM_BANK0 = 0xA00000
NVRAM_MAGIC = b"nvram"
VZ_AUX_HEADER = 0x4000

ASIF_MAGIC = b"shdw"

AVPBOOTER_IN_GUEST = ("System/Library/Frameworks/Virtualization.framework/"
                      "Versions/A/Resources/AVPBooter.vmapple2.bin")


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def strip_aux_header(raw):
    def bank_at(off):
        return raw[off:off + 64].find(NVRAM_MAGIC) >= 0

    if bank_at(NVRAM_BANK0):
        return raw
    if bank_at(NVRAM_BANK0 + VZ_AUX_HEADER):
        return raw[VZ_AUX_HEADER:]
    sys.exit("auxiliary storage has no CHRP NVRAM bank at either known offset; "
             "refusing to guess its layout")


def vm_running():
    """Whether any Virtualization.framework VM is up on this Mac."""
    return subprocess.run(["pgrep", "-f", "com.apple.Virtualization.VirtualMachine"],
                          capture_output=True).returncode == 0


def clone(src, dst):
    """APFS clonefile when possible (instant, no space), a copy otherwise."""
    if os.path.exists(dst):
        os.remove(dst)
    if subprocess.run(["cp", "-c", src, dst], capture_output=True).returncode == 0:
        return "cloned"
    shutil.copyfile(src, dst)
    return "copied"


def is_asif(path):
    with open(path, "rb") as f:
        return f.read(4) == ASIF_MAGIC


def attach(image, asif):
    """Attach read-only and unmounted; the whole-disk node, e.g. 'disk13'.

    ASIF only attaches through `diskutil image`; hdiutil refuses the format. A
    raw file has no header to recognise it by, so hdiutil is told what it is —
    forcing that class on an ASIF, conversely, attaches its container bytes as
    if they were the disk, and no partition table is found.
    """
    if asif:
        out = run(["diskutil", "image", "attach", "--readOnly", "--noMount", image]).stdout
        nodes = [l.split()[0] for l in out.splitlines() if l.startswith("/dev/disk")]
    else:
        info = plistlib.loads(run(["hdiutil", "attach", "-readonly", "-nomount", "-plist",
                                   "-imagekey", "diskimage-class=CRawDiskImage",
                                   image]).stdout.encode())
        nodes = [e["dev-entry"] for e in info["system-entities"]]
    whole = [n for n in nodes if n.rstrip("0123456789").endswith("disk")]
    if not whole:
        raise RuntimeError(f"attaching {image} gave no whole-disk node")
    return whole[0].split("/")[-1]


def guest_avpbooter(whole, dest):
    """Copy AVPBooter out of the guest's sealed system volume, read-only."""
    apfs = plistlib.loads(run(["diskutil", "apfs", "list", "-plist"]).stdout.encode())
    # The APFS containers on the image are synthesized as disks of their own;
    # the one to open is the volume whose role is System.
    system = [v["DeviceIdentifier"]
              for c in apfs.get("Containers", [])
              if any(p["DeviceIdentifier"].startswith(whole + "s")
                     for p in c.get("PhysicalStores", []))
              for v in c.get("Volumes", []) if "System" in v.get("Roles", [])]
    if not system:
        raise RuntimeError("no APFS volume with the System role on this disk")
    mount = tempfile.mkdtemp(prefix="orchard-guest-")
    try:
        run(["diskutil", "mount", "readOnly", "nobrowse", "-mountPoint", mount, system[0]])
        try:
            shutil.copyfile(os.path.join(mount, AVPBOOTER_IN_GUEST), dest)
        finally:
            subprocess.run(["diskutil", "unmount", system[0]], capture_output=True)
    finally:
        os.rmdir(mount)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("utm", help="the .utm bundle (UTM keeps them in "
                    "~/Library/Containers/com.utmapp.UTM/Data/Documents)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "images"),
                    help="where the files land (default: images/)")
    ap.add_argument("--boot-args", default="-v serial=3",
                    help="NVRAM boot-args to add; '' for none (default: %(default)r)")
    ap.add_argument("--rom", help="use this AVPBooter.vmapple2.bin instead of "
                    "taking the guest's own")
    ap.add_argument("--disk-format", choices=("qcow2", "raw"), default="qcow2",
                    help="format of the copied system disk (default: %(default)s)")
    ap.add_argument("--force", action="store_true",
                    help="convert even though a VM appears to be running")
    args = ap.parse_args()

    cfg = plistlib.load(open(os.path.join(args.utm, "config.plist"), "rb"))
    if cfg.get("Backend") != "Apple":
        sys.exit(f"{args.utm} is not a Virtualization.framework VM (Backend="
                 f"{cfg.get('Backend')!r}); only UTM's Virtualize backend makes one")
    if vm_running() and not args.force:
        sys.exit("a virtual machine is running: shut it down first, or the disk is "
                 "cloned halfway through a write (--force to do it anyway)")

    platform = cfg["System"]["MacPlatform"]
    drives = cfg.get("Drive", [])
    if len(drives) != 1:
        sys.exit(f"expected one drive, found {len(drives)}; pass the system disk's VM")
    data = os.path.join(args.utm, "Data")
    src_disk = os.path.join(data, drives[0]["ImageName"])
    src_aux = os.path.join(data, platform.get("AuxiliaryStoragePath", "AuxiliaryStorage"))

    out = os.path.abspath(args.out_dir)
    os.makedirs(out, exist_ok=True)
    disk_name = f"disk.{args.disk_format}"
    disk = os.path.join(out, disk_name)
    aux = os.path.join(out, "aux.img")
    # The stock firmware is only an input to the patch; it does not stay.
    rom_raw = os.path.join(tempfile.mkdtemp(prefix="orchard-rom-"), "AVPBooter.vmapple2.bin")
    rom = os.path.join(out, "AVPBooter.patched.bin")
    overlay = os.path.join(out, "overlay.qcow2")

    asif = is_asif(src_disk)
    whole = attach(src_disk, asif)
    try:
        if asif or args.disk_format != "raw":
            if os.path.exists(disk):
                os.remove(disk)
            print(f"{disk_name:<12} writing from {src_disk} (minutes; zeros are skipped)...",
                  flush=True)
            # -m 1: one request in flight, in order. Out-of-order writes are
            # what makes a USB drive crawl.
            subprocess.run(["qemu-img", "convert", "-p", "-m", "1", "-f", "raw",
                            "-O", args.disk_format, f"/dev/r{whole}", disk], check=True)
            how = "written"
        else:
            how = clone(src_disk, disk)
        st = os.stat(disk)
        print(f"{disk_name:<12} {how}: {st.st_blocks * 512 / 2**30:.1f} GiB on disk")

        if args.rom:
            shutil.copyfile(args.rom, rom_raw)
            print(f"AVPBooter    from {args.rom}")
        else:
            try:
                guest_avpbooter(whole, rom_raw)
                print("AVPBooter    taken from the guest's own Virtualization.framework")
            except (subprocess.CalledProcessError, RuntimeError, OSError) as exc:
                detail = getattr(exc, "stderr", "") or exc
                sys.exit(f"could not read AVPBooter out of the guest ({detail}).\n"
                         f"Copy {'/' + AVPBOOTER_IN_GUEST} out of the running guest "
                         f"and pass it with --rom.")
    finally:
        subprocess.run(["hdiutil", "detach", whole], capture_output=True)

    open(aux, "wb").write(strip_aux_header(open(src_aux, "rb").read()))
    print(f"aux.img      {os.path.getsize(aux)} bytes, header stripped")

    ecid = plistlib.loads(platform["MachineIdentifier"])["ECID"]
    display = (cfg.get("Display") or [{}])[0]
    config = {
        "version": 1,
        "os": "darwin",
        "arch": "arm64",
        "diskFormat": "raw",
        "ecid": base64.b64encode(platform["MachineIdentifier"]).decode(),
        "hardwareModel": base64.b64encode(platform["HardwareModel"]).decode(),
        "memorySize": cfg["System"].get("MemorySize", 0) * 2**20,
        "cpuCount": cfg["System"].get("CPUCount", 0),
        "display": {"width": display.get("WidthPixels", 0),
                    "height": display.get("HeightPixels", 0)},
    }
    json.dump(config, open(os.path.join(out, "config.json"), "w"))
    print(f"config.json  ECID {ecid}")

    subprocess.run([sys.executable, os.path.join(HERE, "patch-avpbooter.py"), rom_raw, rom],
                   check=True)
    shutil.rmtree(os.path.dirname(rom_raw), ignore_errors=True)

    if args.boot_args:
        subprocess.run([sys.executable, os.path.join(HERE, "prepare-aux.py"), aux,
                        "--set", f"boot-args={args.boot_args}"], check=True)

    # Relative backing path: resolved against the overlay's own folder, so the
    # pair keeps working wherever it is copied, the phone included.
    if os.path.exists(overlay):
        os.remove(overlay)
    run(["qemu-img", "create", "-q", "-f", "qcow2", "-F", args.disk_format, "-b", disk_name,
         "overlay.qcow2"], cwd=out)
    print(f"overlay.qcow2 over {disk_name}")

    print(f"\n{out} is ready:")
    for name in (disk_name, "overlay.qcow2", "aux.img", "AVPBooter.patched.bin", "config.json"):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
