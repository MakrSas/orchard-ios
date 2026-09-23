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
#   qemu/build-ios        meson setup against a cross-file made from
#                         scripts/cross-ios-arm64.txt.in, which also selects
#                         configs/devices/aarch64-softmmu/ios.mak.
#
# The C dependencies (glib, pixman, libslirp, ...) come prebuilt: deps/ios, as
# scripts/fetch-ios-deps.sh unpacks them, or wherever ORCHARD_IOS_DEPS points.
# The Rust display device is built by meson itself, with cargo, for
# aarch64-apple-ios and backend-metal.
#
# Needs Xcode with the iPhoneOS SDK, rustup, ninja and pkg-config.
#
# Re-running is cheap: each tree is only configured when it does not exist yet,
# and ninja rebuilds what changed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QEMU="$ROOT/qemu"
BOOT="$QEMU/build-bootstrap"
IOS="${BUILD_DIR:-$QEMU/build-ios}"
DEPS="${ORCHARD_IOS_DEPS:-$ROOT/deps/ios}"
JOBS="${JOBS:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.ncpu)}"

[ -f "$DEPS/lib/libglib-2.0.a" ] || {
    echo "No iOS dependencies in $DEPS. Run scripts/fetch-ios-deps.sh or set ORCHARD_IOS_DEPS=" >&2
    exit 1
}
DEPS="$(cd "$DEPS" && pwd)"

# The cross-file, filled in for this Mac. Meson reads a cross-file's compiler
# flags only when a build tree is first set up — not on reconfigure — so when
# it changes (another Xcode, other flags) the tree is set up again from clean.
sdk="$(xcrun --sdk iphoneos --show-sdk-path)"
toolchain="$(dirname "$(dirname "$(dirname "$(xcrun --find ar)")")")"
mkdir -p "$IOS"
CROSS="$IOS.cross.txt"
sed -e "s|@SDK@|$sdk|g" -e "s|@TOOLCHAIN@|$toolchain|g" -e "s|@DEPS@|$DEPS|g" \
    "$ROOT/scripts/cross-ios-arm64.txt.in" > "$CROSS.new"
wipe=()
if cmp -s "$CROSS.new" "$CROSS"; then
    rm "$CROSS.new"
else
    mv "$CROSS.new" "$CROSS"
    if [ -f "$IOS/build.ninja" ]; then
        echo "==> the cross-file changed: setting the build tree up again"
        wipe=(--wipe)
    fi
fi
rustup target list --installed | grep -qx aarch64-apple-ios || rustup target add aarch64-apple-ios

if [ ! -f "$BOOT/config-host.mak" ]; then
    echo "==> bootstrap configure (native)"
    mkdir -p "$BOOT"
    (cd "$BOOT" && ../configure --target-list=aarch64-softmmu --disable-docs \
        --enable-slirp --disable-werror --with-devices-aarch64=ios)
fi

if [ ! -f "$IOS/build.ninja" ] || [ ${#wipe[@]} -gt 0 ]; then
    echo "==> meson setup for iOS"
    cp "$BOOT/config-host.mak" "$IOS/config-host.mak"
    (cd "$IOS" && "$BOOT/pyvenv/bin/meson" setup ${wipe[@]+"${wipe[@]}"} . .. \
        --cross-file="$CROSS" \
        -Dbuildtype=release -Dprefix="$DEPS" \
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
