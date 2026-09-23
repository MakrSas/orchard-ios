#!/usr/bin/env bash
# Build libqemu-aarch64-softmmu.dylib for an arm64 iOS device, the library
# ios/app loads with dlopen.
#
# Two build trees, because this QEMU still has its configure script and the
# iOS tree cannot run it:
#
#   qemu/build-bootstrap  configure, run natively once. Its only job is the
#                         Python venv with a matching meson and a
#                         config-host.mak (TARGET_DIRS and friends): meson.build
#                         reads that file but only configure writes it.
#   qemu/build-ios        meson setup against scripts/cross-ios-arm64.txt, which
#                         also selects configs/devices/aarch64-softmmu/ios.mak.
#
# The C dependencies (glib, pixman, libslirp, ...) come prebuilt from
# ~/inferno-ios/prefix, the same prefix Inferno-iOS builds against; see the
# cross-file. The Rust display device is built by meson itself, with cargo, for
# aarch64-apple-ios and backend-metal.
#
# Re-running is cheap: each tree is only configured when it does not exist yet,
# and ninja rebuilds what changed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QEMU="$ROOT/qemu"
BOOT="$QEMU/build-bootstrap"
IOS="$QEMU/build-ios"
CROSS="$ROOT/scripts/cross-ios-arm64.txt"
JOBS="${JOBS:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.ncpu)}"

sdk="$(xcrun --sdk iphoneos --show-sdk-path)"
if ! grep -q "$sdk" "$CROSS"; then
    echo "warning: $CROSS names an SDK other than $sdk" >&2
    echo "         update its -isysroot paths if the build cannot find headers" >&2
fi
rustup target list --installed | grep -qx aarch64-apple-ios || rustup target add aarch64-apple-ios

if [ ! -f "$BOOT/config-host.mak" ]; then
    echo "==> bootstrap configure (native)"
    mkdir -p "$BOOT"
    (cd "$BOOT" && ../configure --target-list=aarch64-softmmu --disable-docs \
        --enable-slirp --disable-werror --with-devices-aarch64=ios)
fi

if [ ! -f "$IOS/build.ninja" ]; then
    echo "==> meson setup for iOS"
    mkdir -p "$IOS"
    cp "$BOOT/config-host.mak" "$IOS/config-host.mak"
    (cd "$IOS" && "$BOOT/pyvenv/bin/meson" setup . .. \
        --cross-file="$CROSS" \
        -Dbuildtype=release -Dprefix="$HOME/inferno-ios/prefix" \
        -Dshared_lib=true -Db_staticpic=true -Dwerror=false \
        -Dkvm=disabled -Dhvf=disabled -Dwhpx=disabled \
        -Dcocoa=disabled -Dgtk=disabled -Dsdl=disabled -Dcurses=disabled \
        -Dcoreaudio=disabled -Dcurl=disabled \
        -Dvnc=enabled -Dvnc_jpeg=disabled -Dvnc_sasl=disabled \
        -Dtools=disabled -Dlibssh=disabled -Dbzip2=disabled \
        -Dcoroutine_backend=ucontext -Dqom_cast_debug=false)
fi

echo "==> ninja -j$JOBS"
"${NINJA:-ninja}" -C "$IOS" -j"$JOBS"

lib="$IOS/libqemu-aarch64-softmmu.dylib"
echo
ls -la "$lib"
vtool -show-build-version "$lib" | grep -E 'platform|minos'
# Read once into a variable: `nm | grep -q` under pipefail reports a miss for
# every symbol, since grep's early exit kills nm with SIGPIPE.
exported="$(nm -gU "$lib")"
for sym in qemu_init qemu_main_loop qemu_cleanup orchard_display_attach orchard_display_read \
           orchard_input_touch orchard_input_function_key orchard_input_key_tap orchard_input_pointer reims_vgpu_qemu_scanout_copy; do
    if grep -q " _$sym\$" <<<"$exported"; then echo "  exports $sym"; else echo "  MISSING $sym"; fi
done
