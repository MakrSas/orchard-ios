#!/usr/bin/env bash
# stop-previous-vm.sh — kill the running reims-vgpu QEMU and wait until the port
# it holds is actually free.
#
# Every harness here starts by killing the previous boot and then sleeping a
# fixed few seconds. That sleep is a guess, and `AGENTS.md` already describes
# what happens when it is short: the old QEMU still holds `localhost:2222`, the
# new boot dies on
#
#     Could not set up host forwarding rule 'tcp::2222-:22'
#
# and every other line of its output looks like a normal start. The rail is then
# reported NO-BOOT, or — worse, if a stale VM survives — `ssh macos-vm` answers
# from the *old* guest and the probe drives the previous build.
#
# Five seconds is enough for a healthy QEMU and is not enough for the case that
# matters: a macos-11 boot that has just taken a GPU reset is inside device
# recreation and takes longer to die. That is exactly the boot a sweep follows
# with another one, so the fixed sleep fails precisely on the rail being
# investigated. It cost macos-12 a whole leg of a four-rail sweep.
#
# So this waits for the condition instead of for the clock: the process gone
# *and* nothing listening on the port. It escalates to SIGKILL rather than
# waiting forever, and it says which of the two it was still waiting on.
#
# Usage:
#   scripts/app-sweep-probe/stop-previous-vm.sh [--port 2222] [--timeout 60]
set -uo pipefail
export LC_ALL=C

PORT=2222
TIMEOUT=60
# One character bracketed, so the pattern does not match the shell running it.
PATTERN='qemu-system-x86_6[4].*reims-vgpu'

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "stop-previous-vm: unknown argument $1" >&2; exit 2 ;;
  esac
done

say() { echo "stop-previous-vm: $*"; }
alive() { pgrep -f "$PATTERN" >/dev/null 2>&1; }
port_held() { ss -Hltn "sport = :$PORT" 2>/dev/null | grep -q .; }

alive && pkill -f "$PATTERN" 2>/dev/null

# SIGKILL at the halfway mark: a QEMU that has not gone by then is wedged, and
# waiting the rest of the budget for it only shortens what is left for the port.
escalate=$((TIMEOUT / 2))
waited=0
while [ "$waited" -lt "$TIMEOUT" ]; do
  if ! alive && ! port_held; then
    [ "$waited" -gt 0 ] && say "previous VM gone and :$PORT free after ${waited}s"
    exit 0
  fi
  if [ "$waited" -eq "$escalate" ] && alive; then
    say "still alive after ${waited}s — SIGKILL"
    pkill -9 -f "$PATTERN" 2>/dev/null
  fi
  sleep 1
  waited=$((waited + 1))
done

# Which of the two it was is the whole diagnostic value of the line.
alive && say "gave up after ${TIMEOUT}s: QEMU still running"
port_held && say "gave up after ${TIMEOUT}s: something still listens on :$PORT"
exit 1
