#!/bin/bash
# Builds OrchardJITProbe.ipa — nothing but the JIT smoke test. No QEMU dylib,
# no guest tools: this exists to answer one question (does MAP_JIT / the
# self-trace fallback actually work in this repo, under this signing setup)
# before the much bigger job of cross-compiling Orchard's QEMU for iOS.
set -euo pipefail

unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="$ROOT/.build"
IPA="$ROOT/OrchardJITProbe.ipa"
APP="$BUILD/Payload/OrchardJITProbe.app"
SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
TARGET="arm64-apple-ios16.0"

rm -rf "$BUILD"
mkdir -p "$APP"

echo "==> Компиляция"
xcrun --sdk iphoneos swiftc \
    -target "$TARGET" \
    -sdk "$SDK" \
    -O -parse-as-library \
    -o "$APP/OrchardJITProbe" \
    "$ROOT/Sources/JITProbeApp.swift" \
    "$ROOT/../app/Sources/JIT.swift" \
    "$ROOT/../app/Sources/LogCapture.swift" \
    "$ROOT/../app/Sources/L10n.swift"

echo "==> Сборка бандла"
cp "$ROOT/Resources/Info.plist" "$APP/Info.plist"
chmod +x "$APP/OrchardJITProbe"

echo "==> Подпись"
codesign --force --sign - --timestamp=none \
    --entitlements "$ROOT/Resources/entitlements.plist" \
    "$APP"

echo "==> Упаковка"
cd "$BUILD"
rm -f "$IPA"
zip -qry "$IPA" Payload -x '*.DS_Store'

echo
echo "Готово: $IPA"
ls -lh "$IPA"
