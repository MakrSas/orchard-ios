#!/usr/bin/env bash
# desktop-rate.sh — how often does this rail reach a desktop, and when it does
# not, what did the guest's window server say?
#
# A rail that fails a third of its boots cannot be judged by one boot, in either
# direction — the same rule `AGENTS.md` states for guest kernel panics. This
# boots one rail N times and reports the rate, and it separates the two failures
# that a `pgrep -x Dock` loop cannot tell apart:
#
# - **The guest is still booting.** Nothing is wrong; the wait was short.
# - **WindowServer aborted.** On macos-11 this happens ~13 s into boot and the
#   screen keeps showing the verbose kernel console, because there is no window
#   server to draw anything else. `ssh` answers, `guest-authorize.sh` installs
#   its key, the device presents frames at ~29 Hz, and every signal a harness
#   reads says the guest is healthy. It is: nobody is logged in and nothing is
#   composited. The evidence is a crash report the guest writes itself, whose
#   `Application Specific Information` names the failure:
#
#       IOConnectMapMemory VRAM failed: port 0x… error 0xe00002c2
#       Failed to create FB 1 of 1 (Failed to map VRAM)
#
#   `0xe00002c2` is `kIOReturnBadArgument` — `AppleParavirtFramebuffer` refused
#   CoreDisplay a mappable aperture, so this is a statement about this device and
#   not about the guest.
#
# Each boot keeps its own device fail log and, when there is one, the crash
# report. Those two files side by side across a failing and a passing boot of the
# same binary are the whole point of the script: the difference between them is
# the bug.
#
# Usage:
#   scripts/app-sweep-probe/desktop-rate.sh [--rail NAME] [--boots N] [--out DIR]
set -uo pipefail
export LC_ALL=C

RAIL="macos-11"
BOOTS=6
OUT="/tmp/reims-desktop-rate"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --rail) RAIL="$2"; shift 2 ;;
    --boots) BOOTS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,34p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "desktop-rate: unknown argument $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT"
say() { echo "desktop-rate: $*"; }
RESULTS="$OUT/results.tsv"
: >"$RESULTS"
gssh() { timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=5 macos-vm "$1" 2>/dev/null; }

for n in $(seq 1 "$BOOTS"); do
  say "=== $RAIL boot $n of $BOOTS ==="
  # Waits on the port rather than a clock, so a boot is never scored NO-BOOT for
  # losing the hostfwd race with its own predecessor.
  "$REPO/scripts/app-sweep-probe/stop-previous-vm.sh" || \
    say "boot $n: previous VM still holds :2222"
  rm -f /tmp/reims-vgpu-fail.log
  BOOTLOG="$OUT/$n-boot.log"
  TESTING_TIMEOUT=900 nohup "$REPO/vm/boot-x86.sh" --device reims-vgpu-pci \
    --rail "$RAIL" --testing >"$BOOTLOG" 2>&1 &

  # Only a live device writes a fail log, so this cannot be answered by a
  # surviving QEMU from the previous round.
  up=no
  for _ in $(seq 1 60); do
    [ -f /tmp/reims-vgpu-fail.log ] && { up=yes; break; }
    sleep 5
  done
  [ "$up" = yes ] || { say "boot $n: no device"; printf '%s\tNO-BOOT\t-\t-\n' "$n" >>"$RESULTS"; continue; }

  timeout 300 "$REPO/vm/guest-authorize.sh" >"$OUT/$n-authorize.log" 2>&1

  # WindowServer aborts ~13 s in when it aborts at all, so a verdict is
  # available long before the Dock would have appeared. Both are watched: the
  # crash is the interesting answer and the Dock is the passing one.
  # Three outcomes, and a bare `pgrep -x Dock` loop reports all three as one.
  # `wait-for-desktop.sh` handles the login window — macos-12 stops there on
  # about half its boots — so a rail that only needed a password is not scored as
  # a rail that failed. What is left after that is the interesting failure.
  verdict=TIMEOUT
  "$REPO/scripts/app-sweep-probe/wait-for-desktop.sh" --timeout 400 \
    >"$OUT/$n-desktop-wait.log" 2>&1 && verdict=DESKTOP
  if [ "$verdict" != DESKTOP ] \
    && [ -n "$(gssh 'ls /Library/Logs/DiagnosticReports/WindowServer*.crash 2>/dev/null')" ]; then
    verdict=WS-CRASH
  fi

  # Kept per boot and named by verdict, because the pair is the measurement.
  cp -f /tmp/reims-vgpu-fail.log "$OUT/$n-$verdict-fail.log" 2>/dev/null
  crash=""
  if [ "$verdict" = WS-CRASH ]; then
    crash=$(gssh 'ls -t /Library/Logs/DiagnosticReports/WindowServer*.crash 2>/dev/null | head -1')
    [ -n "$crash" ] && gssh "cat '$crash'" >"$OUT/$n-windowserver.crash"
    # The one line that says which contract was refused.
    grep -m1 'IOConnectMapMemory' "$OUT/$n-windowserver.crash" 2>/dev/null \
      | sed "s/^/desktop-rate: boot $n: /"
  fi

  panic=$(grep -qF 'guest kernel panic' "$BOOTLOG" && echo PANIC || echo no-panic)
  printf '%s\t%s\t%s\t%s\n' "$n" "$verdict" "$panic" "${crash:-none}" >>"$RESULTS"
  say "boot $n: $verdict $panic"
done

pkill -f 'qemu-system-x86_6[4].*reims-vgpu' 2>/dev/null

echo
echo "=== $RAIL over $BOOTS boots ==="
cat "$RESULTS"
echo
awk -F'\t' '{c[$2]++} END {for (v in c) printf "%-10s %d\n", v, c[v]}' "$RESULTS"
