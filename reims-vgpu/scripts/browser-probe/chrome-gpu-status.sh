#!/usr/bin/env bash
#
# scripts/browser-probe/chrome-gpu-status.sh — ask Chrome in the guest what it
# thinks of the GPU, in text, with a profile that has no history.
#
# Chrome latches "GPU access is disabled due to frequent crashes" into the user
# data dir, so a reused profile reports the verdict of some earlier boot rather
# than of the build under test. Every run here gets a fresh --user-data-dir for
# that reason; the cost is one cold GPU-process start per run.
#
# Extra Chrome flags are passed through, which is how the ANGLE backend is
# varied (--use-angle=metal|gl|swiftshader) without editing this script.
#
#   scripts/browser-probe/chrome-gpu-status.sh [--label NAME] [-- <chrome flags>]
#
# Prints the Graphics Feature Status block, the GL/driver identity lines and the
# Problems Detected block. Requires the guest reachable as ssh host `macos-vm`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUEST="${GUEST:-macos-vm}"
PORT="${CDP_PORT:-9222}"
LABEL="run"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done
EXTRA=("$@")

CHROME='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
PROFILE="/tmp/chrome-probe-$LABEL"

scp -q "$SCRIPT_DIR/cdp.py" "$SCRIPT_DIR/chrome_gpu_report.py" "$GUEST:/tmp/"

# `--no-first-run` alone still lets the updater and the default-browser prompt
# start; both add minutes to a cold profile and neither is under test.
ssh "$GUEST" "pkill -f 'Google Chrome' >/dev/null 2>&1; sleep 2; rm -rf '$PROFILE'
nohup '$CHROME' --user-data-dir='$PROFILE' --no-first-run --no-default-browser-check \
  --disable-features=ChromeWhatsNewUI --remote-debugging-port=$PORT \
  ${EXTRA[*]:-} about:blank >/tmp/chrome-$LABEL.out 2>&1 &
sleep 12"

ssh "$GUEST" "python3 /tmp/chrome_gpu_report.py"
