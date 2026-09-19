#!/usr/bin/env bash
# Boot and leave the guest alone.
#
# macOS finishes install/update sequences ACROSS reboots and records progress in
# NVRAM (the aux) and on disk (the overlay). Copying frozen baselines over
# either before a boot wipes that progress, so the guest retries phase 1 forever
# and every session looks like a permanent reboot loop. This script therefore
# resets nothing: expect 2-5 restarts on a fresh image, then convergence.
#
# It also refuses to start a second VM against the same images, because two
# QEMUs sharing one overlay corrupt it.
#
# Copyright (c) 2026 Youssef Elliethy (yaelliethy)
# SPDX-License-Identifier: GPL-2.0-or-later
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# `pgrep -x` cannot match a name this long, and plain `pgrep -f` also matches any
# shell whose own arguments contain the pattern - including the one running this
# script. `pidof` matches the executable, which is the question being asked.
if pidof -q qemu-system-aarch64; then
  echo "a VM is already running against these images; not starting a second one" >&2
  exit 1
fi

rm -f "$ROOT/serial.txt"

# Host GPU knobs that are measured wins on RADV:
#   nodcc  - DCC on a render target the device also reads back costs more than
#            it saves here, and hid a class of stale-tile bugs.
#   nohiz  - likewise for HiZ on depth the guest never samples.
export RADV_DEBUG="${RADV_DEBUG:-nodcc,nohiz}"

nohup "$ROOT/scripts/run-vm.sh" > "$ROOT/vm.log" 2>&1 < /dev/null &
echo "booted; serial -> $ROOT/serial.txt, device log -> $ROOT/vm.log"
