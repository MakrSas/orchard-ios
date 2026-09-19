#!/usr/bin/env bash
# Build QEMU (with the vmapple + reims-vgpu changes) and the Rust GPU device.
#
# The Rust crate is built by QEMU's own meson rule, so this script only has to
# configure QEMU and run ninja; `REIMS_VGPU_DIR` tells that rule where the
# crate is.
#
# Copyright (c) 2026 Youssef Elliethy (yaelliethy)
# SPDX-License-Identifier: GPL-2.0-or-later
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
export REIMS_VGPU_DIR="${REIMS_VGPU_DIR:-$ROOT/reims-vgpu}"

# QEMU builds a Python venv for its own tooling. If this host has an old
# `packaging` in ~/.local, it shadows the venv's copy and the install of
# QEMU's python package dies with
#   TypeError: canonicalize_version() got an unexpected keyword argument
# Disabling the user site directory is enough and affects nothing else.
export PYTHONNOUSERSITE=1

# The GPU is a submodule, pinned to the upstream commit this tree was tested
# against, and kept pristine: our changes to it live in patches/reims-vgpu/ so
# the boundary between steelbrain's work and ours stays visible. Apply them
# here, skipping any that is already in.
if [ -d "$ROOT/.git" ] && [ ! -f "$REIMS_VGPU_DIR/Cargo.toml" ]; then
  git -C "$ROOT" submodule update --init --recursive
fi
# `-e`, not `-d`: in a submodule `.git` is a file holding a gitdir pointer, and
# testing for a directory silently skips every patch.
if git -C "$REIMS_VGPU_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  for patch in "$ROOT"/patches/reims-vgpu/*.patch; do
    [ -e "$patch" ] || continue
    if git -C "$REIMS_VGPU_DIR" apply --reverse --check "$patch" 2>/dev/null; then
      continue                              # already applied
    fi
    echo "applying $(basename "$patch")"
    git -C "$REIMS_VGPU_DIR" apply "$patch"
  done
fi

cd "$ROOT/qemu"
mkdir -p build
cd build
if [ ! -f build.ninja ]; then
  ../configure --target-list=aarch64-softmmu --disable-docs --enable-slirp \
               --disable-werror "$@"
fi
ninja qemu-system-aarch64
echo
echo "built: $ROOT/qemu/build/qemu-system-aarch64"
echo "device: $REIMS_VGPU_DIR/target/release/libreims_vgpu.a"
