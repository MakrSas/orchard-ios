#!/usr/bin/env bash
# Maintainers only: pack an install prefix of the iOS C dependencies into the
# archive scripts/fetch-ios-deps.sh downloads, for upload as a release asset.
#
#   scripts/pack-ios-deps.sh /path/to/prefix [out.tar.xz]
#
# Only what linking needs goes in: headers, static libraries, pkg-config files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="$(cd "${1:?usage: $0 PREFIX [OUT]}" && pwd)"
OUT="${2:-$PWD/orchard-ios-deps-arm64.tar.xz}"

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
rsync -a --include='*/' --include='*.a' --include='*.pc' --include='*.h' \
    --exclude='*' --prune-empty-dirs "$PREFIX/lib" "$stage/"
rsync -a "$PREFIX/include" "$stage/"
echo "$PREFIX" > "$stage/PREFIX"
cp "$ROOT/scripts/ios-deps-SOURCES.md" "$stage/SOURCES.md"
tar -cJf "$OUT" -C "$stage" .
ls -la "$OUT"
