#!/bin/bash
# Makes a macOS guest cheaper to emulate. Run it inside the guest, once, while
# the VM still runs somewhere fast (UTM on the Mac it came from), then convert
# it again with scripts/utm-to-orchard.py.
#
# Under TCG on a phone every instruction the guest runs is paid for, and so is
# every frame it composites. None of this changes what the guest can do; it
# stops the work nobody is waiting for: indexing, animation, blur, background
# checks.
#
#   bash guest-tune.sh          (asks for the admin password once, for sudo)
#
# Each step says whether it took. A step that fails leaves the rest running.
set -u

ok()   { printf '  ok    %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; }
step() { local what="$1"; shift; if "$@" >/dev/null 2>&1; then ok "$what"; else fail "$what"; fi; }

echo "==> system (sudo)"
sudo -v || { echo "sudo is needed for the system half"; exit 1; }
# Spotlight indexes the whole disk after every boot that finds it changed,
# which on a fresh conversion is every boot.
step "Spotlight indexing off"          sudo mdutil -a -i off
step "automatic update checks off"     sudo softwareupdate --schedule off
# Sleep would stop the guest's clock mid-boot; nothing here wants it.
step "no sleep, no display sleep"      sudo pmset -a sleep 0 displaysleep 0 disksleep 0
step "no Time Machine schedule"        sudo tmutil disable

echo "==> this user"
# Blur and motion are the compositor's most expensive work. These two keys may
# need Full Disk Access for Terminal; if they fail, set them by hand in
# System Settings → Accessibility → Display.
step "reduce transparency"             defaults write com.apple.universalaccess reduceTransparency -bool true
step "reduce motion"                   defaults write com.apple.universalaccess reduceMotion -bool true
step "no window open/close animation"  defaults write NSGlobalDomain NSAutomaticWindowAnimationsEnabled -bool false
step "instant window resize"           defaults write NSGlobalDomain NSWindowResizeTime -float 0.001
step "no Dock launch bounce"           defaults write com.apple.dock launchanim -bool false
step "Dock minimise: scale"            defaults write com.apple.dock mineffect -string scale
step "no Mission Control animation"    defaults write com.apple.dock expose-animation-duration -float 0
step "no screen saver"                 defaults -currentHost write com.apple.screensaver idleTime -int 0
step "Siri off"                        defaults write com.apple.assistant.support "Assistant Enabled" -bool false
killall Dock >/dev/null 2>&1

cat <<'EOF'

Left to do by hand, in System Settings:
  - Users & Groups → Automatically log in as: this user (needs FileVault off).
    The phone then goes straight to the desktop without typing a password.
  - General → Login Items: remove everything under "Open at Login", and turn
    off what "Allow in the Background" lists that you do not need.
  - Wallpaper: a plain colour instead of a dynamic or photo one.

Then shut the guest down (not restart), convert it again with
scripts/utm-to-orchard.py, and replace the whole OrchardVM folder on the
phone — the old overlay.qcow2 belongs to the old disk and must not be kept.
EOF
