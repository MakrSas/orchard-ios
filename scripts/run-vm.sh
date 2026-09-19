#!/usr/bin/env bash
# Boot macOS on QEMU's `apple-vm` machine through the real Apple chain:
# AVPBooter -> iBootStage1 (LLB) -> iBootStage2 (iBoot) -> stock XNU.
#
# Everything below was measured on the reference host (Linux, AMD RENOIR,
# Mesa RADV). The notes say why each choice is what it is, because most of
# them cost a boot to find.
#
# Copyright (c) 2026 Youssef Elliethy (yaelliethy)
# SPDX-License-Identifier: GPL-2.0-or-later
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

QEMU="${QEMU:-$ROOT/qemu/build/qemu-system-aarch64}"
IMAGES="${IMAGES:-$ROOT/images}"
DISK="${DISK:-$IMAGES/disk.raw}"          # tart image, read-only
OVERLAY="${OVERLAY:-$IMAGES/overlay.qcow2}"
AUX="${AUX:-$IMAGES/aux.img}"             # tart NVRAM + LocalPolicy
ROM="${ROM:-$IMAGES/AVPBooter.patched.bin}"
ECID="${ECID:-}"                          # from images/config.json
# 10G, not more: the whole RAMBlock is imported into one Vulkan heap, and this
# class of host tops out around 10.5 GiB of importable memory. Past that the
# import is refused (`guest_ram_map_import_exceeds_heap`) and every texture bind
# falls back to a CPU byte loader - slow, with black backgrounds. Measured at
# 10G: `imported_mib=10241`, import kept.
RAM="${RAM:-10G}"
# 12 vCPUs boots to the desktop in ~75 s here. The earlier ceiling of 6 was a
# livelock in XNU's startup pen, and that is fixed by ORCHARD_PAC_INHERIT rather
# than by anything about WFE; measured at 12 with no WFE arm at all.
CPUS="${CPUS:-12}"
SERIAL_LOG="${SERIAL_LOG:-$ROOT/serial.txt}"

for f in "$QEMU" "$DISK" "$AUX" "$ROM"; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
if [ -z "$ECID" ]; then
  ECID=$(python3 - "$IMAGES/config.json" <<'PY'
import base64, json, plistlib, sys
try:
    cfg = json.load(open(sys.argv[1]))
    print(plistlib.loads(base64.b64decode(cfg["ecid"]))["ECID"])
except Exception:
    pass
PY
)
fi
[ -n "$ECID" ] || { echo "no ECID: set ECID=... (see images/config.json)" >&2; exit 1; }

# Guest writes go to an overlay; the tart image stays pristine and read-only.
if [ ! -e "$OVERLAY" ]; then
  qemu-img create -f qcow2 -F raw -b "$DISK" "$OVERLAY" >/dev/null
  echo "created overlay $OVERLAY"
fi

# --- guest CPU --------------------------------------------------------------
# The stock kernel verifies its own signed pointers, so PAC has to be real or it
# dies with "JOP Hash Mismatch Detected". PAC_INHERIT gives a core brought up by
# PSCI CPU_ON the keys firmware would have left it; without it every secondary
# runs keyless and faults on its first authentication.
export ORCHARD_REAL_PAUTH="${ORCHARD_REAL_PAUTH:-1}"
export ORCHARD_PAC_INHERIT="${ORCHARD_PAC_INHERIT:-1}"

# --- host GPU ---------------------------------------------------------------
# An absolute ICD path is dlopen'd directly and cannot be hijacked by
# LD_LIBRARY_PATH, which matters when a second Vulkan stack (linuxbrew, conda)
# is on the search path and fails to load against the system libwayland.
[ -n "${VK_ICD_FILENAMES:-}" ] && export VK_ICD_FILENAMES
export REIMS_VGPU_WINDOW="${REIMS_VGPU_WINDOW:-1}"
# Present the scanout framebuffer as well as the composited rail. This is what
# puts the *early boot* picture on screen -- the Apple logo and progress bar are
# drawn by iBoot and the kernel through the framebuffer, long before the guest's
# graphics stack produces its first composited frame. Without it the window
# stays blank until the desktop appears.
export REIMS_VGPU_FORCE_SCANOUT="${REIMS_VGPU_FORCE_SCANOUT:-1}"

# A Wayland session needs these to be present and sane for the device's own
# window; they are inherited when the script is run from a desktop terminal and
# defaulted here when it is not (e.g. from a service or over ssh).
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export GDK_BACKEND="${GDK_BACKEND:-wayland}"

exec "$QEMU" \
  -M apple-vm,uuid="$ECID" \
  -accel tcg,thread=multi,tb-size=2048 \
  -m "$RAM" -smp "$CPUS" \
  -bios "$ROM" \
  -drive file="$AUX",if=pflash,format=raw \
  -drive file="$DISK",if=pflash,format=raw,readonly=on \
  -drive file="$AUX",if=none,id=aux,format=raw \
  -device vmapple-virtio-blk-pci,variant=aux,drive=aux,share-rw=on \
  -drive file="$OVERLAY",if=none,id=root,format=qcow2,cache=writeback,aio=threads,discard=unmap \
  -device vmapple-virtio-blk-pci,variant=root,drive=root \
  -netdev user,id=net0 \
  -device virtio-net-pci,netdev=net0 \
  -chardev socket,id=ser0,path="$ROOT/serial.sock",server=on,wait=off,logfile="$SERIAL_LOG" \
  -serial chardev:ser0 \
  -qmp unix:"$ROOT/qmp.sock",server=on,wait=off \
  -display none \
  "$@"
