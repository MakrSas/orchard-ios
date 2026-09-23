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

# The GPU is vendored in-tree at reims-vgpu/ (upstream:
# https://github.com/steelbrain/reims-vgpu, with our changes baked in),
# so there is nothing to fetch or patch here. Fail early if it is missing.
if [ ! -f "$REIMS_VGPU_DIR/Cargo.toml" ]; then
  echo "error: REIMS_VGPU_DIR has no Cargo.toml: $REIMS_VGPU_DIR" >&2
  exit 1
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
