#!/usr/bin/env bash
# Fetch the prebuilt C dependencies scripts/build-ios.sh links QEMU against,
# and unpack them into deps/ios (or $ORCHARD_IOS_DEPS).
#
# They are static libraries for arm64 iOS 16+: glib, gmp, libffi, libintl,
# lz4, lzfse, nettle, pcre2, pixman, libpng, libslirp, libtasn1, libucontext,
# zlib. Nothing in them is Orchard's; SOURCES.md inside the archive lists
# versions and licences. The archive is a release asset of this repository.
#
# The .pc files name the directory they were built into, so they are
# rewritten here to wherever the archive lands.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS="${ORCHARD_IOS_DEPS:-$ROOT/deps/ios}"
TAG="${ORCHARD_IOS_DEPS_TAG:-ios-deps-1}"
ASSET="orchard-ios-deps-arm64.tar.xz"
REPO="${ORCHARD_REPO:-MakrSas/orchard}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "==> $ASSET ($TAG)"
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --pattern "$ASSET" --dir "$tmp"
else
    curl -fL --retry 3 -o "$tmp/$ASSET" \
        "https://github.com/$REPO/releases/download/$TAG/$ASSET"
fi

rm -rf "$DEPS"
mkdir -p "$DEPS"
tar -xJf "$tmp/$ASSET" -C "$DEPS"
DEPS="$(cd "$DEPS" && pwd)"

old="$(cat "$DEPS/PREFIX")"
for pc in "$DEPS"/lib/pkgconfig/*.pc; do
    [ -L "$pc" ] && continue
    sed -i '' "s|$old|$DEPS|g" "$pc"
done
# zlib is the SDK's; point its .pc at this Mac's SDK.
if [ -f "$DEPS/lib/pkgconfig/zlib.pc" ]; then
    sed -i '' "s|^prefix=.*|prefix=$(xcrun --sdk iphoneos --show-sdk-path)/usr|" "$DEPS/lib/pkgconfig/zlib.pc"
fi
echo "$DEPS" > "$DEPS/PREFIX"
echo "==> $DEPS"
