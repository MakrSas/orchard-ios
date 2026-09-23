#!/usr/bin/env bash
#
# vm/drive-desktop.sh — drive a running x86 guest's desktop from the host.
#
# WHY THIS EXISTS. Almost every reading this project takes is a *rate* — draws
# a second, packets a boot, refusals per desktop — and a guest sitting at an
# idle Finder produces none of them. Every such number this project has ever
# quoted was taken behind "host-driven workload (three rounds of launching and
# quitting five applications over ssh)", and that sentence used to be a shell
# script in `/tmp` that each session re-derived from the reading it was
# chasing. Two sessions re-derived it slightly differently, which is the thing
# a committed harness stops: a rate is only comparable across boots if the
# workload is.
#
# It is host-driven on purpose. Input synthesised inside the guest would be the
# guest exercising itself; `AGENTS.md` asks for host-driven input so that what
# the device sees is what a real user's session sends.
#
# The guest must already be up. `vm/rail-sweep.sh` waits for the first frame
# and then runs this; a human driving one boot runs it by hand after
# `first frame presented` appears.
set -u
cd "$(dirname "$0")/.."

ROUNDS=${ROUNDS:-3}
APPS=${APPS:-"TextEdit Preview Calculator Chess Terminal"}

log() { echo "[drive] $(date +%H:%M:%S) $*"; }

# `PreferredAuthentications=none` on purpose: the answer we are waiting for is
# sshd *refusing* us, which proves it is listening. A successful auth would
# also prove it, and needs the key that `guest-authorize.sh` has not installed
# yet.
log "waiting for sshd"
for _ in $(seq 1 150); do
  if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=4 \
         -o PreferredAuthentications=none \
         -p 2222 "$(whoami)@127.0.0.1" true 2>&1 |
       grep -q "Permission denied\|Authentication"; then
    break
  fi
  sleep 5
done
log "sshd answering"

./vm/guest-authorize.sh >/tmp/drive-authorize.log 2>&1
log "authorize rc=$?"

SSH="ssh -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o BatchMode=yes \
     -o StrictHostKeyChecking=no -o ConnectTimeout=8 macos-vm"
$SSH true 2>/dev/null || { log "key auth failed"; exit 1; }

# The Dock is the cheapest "the session is really up" signal that does not
# depend on the device: a rail whose WindowServer never came up has no Dock,
# and driving it would produce a workload of failed `open` calls.
for _ in $(seq 1 40); do
  $SSH 'pgrep -x Dock >/dev/null' 2>/dev/null && break
  sleep 5
done
log "Dock up"

for round in $(seq 1 "$ROUNDS"); do
  for app in $APPS; do
    timeout 25 $SSH "open -a '$app'" 2>/dev/null
    sleep 5
  done
  log "round $round open done"
  sleep 12
  for app in $APPS; do
    timeout 25 $SSH "osascript -e 'tell application \"$app\" to quit'" 2>/dev/null
    sleep 3
  done
  log "round $round quit done"
  sleep 8
done
log "workload done"
