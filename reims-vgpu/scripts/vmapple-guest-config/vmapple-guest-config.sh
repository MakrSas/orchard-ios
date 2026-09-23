#!/usr/bin/env bash
#
# scripts/vmapple-guest-config/vmapple-guest-config.sh
#
# Configure the vmapple macOS guest for headless agent-driven testing:
# disable sleep / display-sleep / disk-sleep / powernap / App Nap / screensaver /
# screen-lock, and keep Remote Login on. Applied from the host over SSH; run it
# against an INTERACTIVE (pristine) boot so the settings persist into the golden
# snapshot.
#
# AUTO-LOGIN IS DELIBERATELY NOT CONFIGURED HERE. It must be enabled BY HAND in the
# guest (System Settings -> Users & Groups -> Automatically log in as ...) on a
# fresh image restore. An earlier scripted /etc/kcpassword approach corrupted the
# stored password and broke auto-login (dropped to the login screen); the
# System-Settings toggle is the reliable path. Do not re-add auto-login here.
#
# The guest login/sudo password defaults to the account name (override via GUEST_PW).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GUEST_USER="${GUEST_USER:-macosvm}"
GUEST_PW="${GUEST_PW:-macosvm}"
SSH_PORT="${SSH_PORT:-2222}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/vmapple_guest}"

ssh_guest() {
  ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ConnectTimeout=10 -p "$SSH_PORT" "$GUEST_USER@localhost" "$@"
}

echo "[guest-config] configuring $GUEST_USER@localhost:$SSH_PORT ..."

# --- Sleep / power-saving (needs sudo) ------------------------------------------
ssh_guest "echo '$GUEST_PW' | sudo -S pmset -a sleep 0 displaysleep 0 disksleep 0 powernap 0 womp 0 autopoweroff 0 standby 0" 2>/dev/null
ssh_guest "echo '$GUEST_PW' | sudo -S pmset -a disablesleep 1" 2>/dev/null

# --- Screensaver / App Nap / screen lock ----------------------------------------
ssh_guest "defaults -currentHost write com.apple.screensaver idleTime -int 0"
ssh_guest "defaults write NSGlobalDomain NSAppSleepDisabled -bool YES"
ssh_guest "echo '$GUEST_PW' | sudo -S sysadminctl -screenLock off -password '$GUEST_PW'" 2>/dev/null || true

# --- Remote Login stays on ------------------------------------------------------
ssh_guest "echo '$GUEST_PW' | sudo -S systemsetup -setremotelogin on" 2>/dev/null || true

# --- Verify ---------------------------------------------------------------------
echo "[guest-config] verification:"
ssh_guest "
  echo -n '  remote login: '; echo '$GUEST_PW' | sudo -S systemsetup -getremotelogin 2>/dev/null | sed 's/^/ /'
  echo '  pmset:'; pmset -g custom | grep -iE 'disablesleep|^ *sleep|displaysleep|powernap|standby|autopoweroff|womp' | sed 's/^/   /'
" 2>/dev/null

echo "[guest-config] done."
echo "[guest-config] NOTE: enable auto-login BY HAND (System Settings -> Users & Groups ->"
echo "                     Automatically log in as ${GUEST_USER}) — it is not scripted."
