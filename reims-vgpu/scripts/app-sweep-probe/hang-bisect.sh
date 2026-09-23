#!/usr/bin/env bash
# hang-bisect.sh — which rail of this device hangs the host GPU?
#
# A `VK_ERROR_DEVICE_LOST` here is not a Vulkan-level refusal, it is the host
# kernel resetting the GPU: `i915 … GPU HANG: ecode … context reset due to GPU
# hang`. The device survives it — it recreates and the guest keeps running — but
# every recreate loses the writebacks in flight (`render_store_lost`) and stalls
# the drain for ten to twenty seconds, which is what `app-sweep-probe` reports as
# FREEZE.
#
# The device's own log cannot say which submission hung: by the time anything is
# observable the fence has already failed and the batch that caused it is gone.
# The kernel's `ecode` is a hash of the hanging context's state and is stable per
# cause, so counting hangs per arm is the instrument, and the arms are the
# `REIMS_VGPU_*` narrowing switches — each one turns a rail off without turning
# anything on, which is the only direction an override is allowed to move.
#
# Each arm is one driven boot. The verdict is the number of `GPU HANG` lines the
# kernel logged during it, read from the journal by timestamp so a hang from a
# previous arm cannot be counted twice.
#
# Usage:
#   scripts/app-sweep-probe/hang-bisect.sh [--rail NAME] [--seconds N]
#     [--arms "shipping GUEST_IMPORT=off COMPUTE_GATHER=off ..."] [--rounds N]
#
# **One boot per arm decides nothing.** Five driven macos-11 boots of one binary,
# shipping or near it, logged 2, 13, 6, 2 and 12 kernel `GPU HANG` lines. Every
# single-boot bisect table this script has produced sits inside that spread. Use
# `--rounds` — it interleaves the arms round by round, so a drift in the host's
# own state over an hour lands on both arms rather than on whichever ran second —
# and read `stalls` in preference to `hangs`: a stall is one wait giving up and a
# kernel line is one reset, and several stalls routinely share one reset.
#
# An arm is either the word `shipping` or one or more comma-joined `NAME=value`
# pairs, each exported as `REIMS_VGPU_NAME=value` for that boot. A comma-joined
# arm is how the whole set of narrowing switches gets tested in one boot, which
# is the first question worth asking: if the hang survives every optimization
# being switched off, no optimization causes it and there is nothing to bisect.
#
# Every `REIMS_VGPU_*` variable in the environment is cleared before each arm, so
# an arm cannot inherit the previous one's switch. The names come from the
# environment itself rather than a list kept here, because a list would go stale
# the first time `env.rs` grows a switch and the drift is invisible.
set -uo pipefail
export LC_ALL=C

RAIL="macos-11"
SECONDS_PER_APP=12
ARMS="shipping GUEST_IMPORT=off COMPUTE_GATHER=off COMPUTE_SCATTER=off"
ROUNDS=1
OUT="/tmp/reims-hang-bisect"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --rail) RAIL="$2"; shift 2 ;;
    --seconds) SECONDS_PER_APP="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "hang-bisect: unknown argument $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT"
say() { echo "hang-bisect: $*"; }
RESULTS="$OUT/results.tsv"
: >"$RESULTS"
# Tag to full arm, for the arms whose tag is a position rather than the string.
: >"$OUT/arms.tsv"

# The kernel's own count, bounded to this arm's wall clock. `--since` takes a
# local timestamp, which is what `date` prints, so the two agree without a
# timezone argument.
hangs_since() {
  journalctl -k --since "$1" --no-pager 2>/dev/null | grep -c 'GPU HANG' || true
}

for round in $(seq 1 "$ROUNDS"); do
arm_no=0
for arm in $ARMS; do
  say "=== round $round arm $arm ==="
  # Not a fixed sleep: an arm that just hung the GPU takes longer to die than
  # one that did not, and this bisect exists to run exactly those arms.
  "$REPO/scripts/app-sweep-probe/stop-previous-vm.sh" || \
    say "$arm: previous VM still holds :2222"
  rm -f /tmp/reims-vgpu-fail.log
  # Exported, never passed as argv: a command line naming the arm would be
  # matched by the `pkill` above on the next round.
  for stale in $(env | sed -n 's/^\(REIMS_VGPU_[A-Z0-9_]*\)=.*/\1/p'); do
    unset "$stale"
  done
  if [ "$arm" != shipping ]; then
    # Comma-joined, so one arm can carry every switch at once.
    old_ifs=$IFS; IFS=,
    for pair in $arm; do
      export "REIMS_VGPU_${pair%%=*}=${pair##*=}"
    done
    IFS=$old_ifs
  fi

  started=$(date '+%Y-%m-%d %H:%M:%S')
  # The arm names the files, but the arm this script's own header calls the
  # first question worth asking carries every switch at once, and that string is
  # longer than a filename may be. Past 48 characters the tag becomes the arm's
  # position instead, and `arms.tsv` carries the full string — every failure
  # here is `File name too long` on the boot log, which reads as a broken boot
  # rather than as a name.
  arm_no=$((arm_no + 1))
  if [ "${#arm}" -le 48 ]; then
    tag="r$round-$arm"
  else
    tag="r$round-arm$arm_no"
  fi
  printf '%s\t%s\n' "$tag" "$arm" >>"$OUT/arms.tsv"
  BOOTLOG="$OUT/$tag-boot.log"
  TESTING_TIMEOUT=900 nohup "$REPO/vm/boot-x86.sh" --device reims-vgpu-pci \
    --rail "$RAIL" --testing >"$BOOTLOG" 2>&1 &

  up=no
  for _ in $(seq 1 60); do
    [ -f /tmp/reims-vgpu-fail.log ] && { up=yes; break; }
    sleep 5
  done
  [ "$up" = yes ] || { say "$tag: no device"; printf '%s\tNO-BOOT\t-\n' "$tag" >>"$RESULTS"; continue; }

  timeout 300 "$REPO/vm/guest-authorize.sh" >/dev/null 2>&1
  # Shared with `sweep-rails.sh`: it separates "still booting" from "stopped at
  # the login window" and logs in for the second, which a bare `pgrep -x Dock`
  # loop cannot do. An arm that never reaches a desktop drives nothing, so its
  # hang count would be a measurement of an idle GPU.
  dock=yes
  "$REPO/scripts/app-sweep-probe/wait-for-desktop.sh" --timeout 400 || dock=no
  [ "$dock" = yes ] || { say "$tag: no desktop"; printf '%s\tNO-DESKTOP\t-\n' "$tag" >>"$RESULTS"; continue; }
  sleep 8

  QMP_SOCK="$("$REPO/scripts/qmp/qmp.py" sock)" timeout 700 \
    "$REPO/scripts/app-sweep-probe/app-sweep-probe.sh" --rail "$RAIL" \
    --seconds "$SECONDS_PER_APP" --torture-seconds 30 \
    --keep "$OUT/$tag-work" >"$OUT/$tag-probe.log" 2>&1
  probe=$?

  # Keep the device's own log before the next arm truncates it: `stalls` is read
  # from it, and it is the better-resolved of the two hang metrics — a stall is
  # one wait giving up, a kernel line is one reset, and several stalls routinely
  # share a reset.
  cp -f /tmp/reims-vgpu-fail.log "$OUT/$tag-fail.log" 2>/dev/null || true
  pkill -f 'qemu-system-x86_6[4].*reims-vgpu' 2>/dev/null
  sleep 6
  hangs=$(hangs_since "$started")
  freezes=$(grep -c 'FREEZE' "$OUT/$tag-probe.log" || true)
  stalls=$(grep -c 'reason=sync_exec_lock_hold ' "$OUT/$tag-fail.log" 2>/dev/null || true)
  lost=$(grep -c 'vk_device_lost' "$OUT/$tag-fail.log" 2>/dev/null || true)
  # The panic verdict outranks the probe's: a guest kernel panic can land after
  # the probe has reported success, so an arm is not clean because `probe_exit`
  # is 0.
  panic=no
  grep -q 'guest kernel panic' "$BOOTLOG" 2>/dev/null && panic=YES
  # The capability line says whether the arm took, so an override that was
  # ignored cannot be scored as an arm that ran.
  caps=$(grep -m1 -o 'host_pointer_import=[a-z_]*' "$OUT/$tag-work"/*.log 2>/dev/null | head -1)
  printf '%s\t%s\thangs=%s\tstalls=%s\tlost=%s\tfreezes=%s\tpanic=%s\tprobe_exit=%s\t%s\n' \
    "$round" "$arm" "$hangs" "$stalls" "$lost" "$freezes" "$panic" "$probe" \
    "${caps:-caps=?}" >>"$RESULTS"
  say "$tag: GPU hangs=$hangs stalls=$stalls lost=$lost freezes=$freezes panic=$panic probe_exit=$probe ${caps:-}"
done
done

echo
echo "=== hang bisect on $RAIL ==="
cat "$RESULTS"
