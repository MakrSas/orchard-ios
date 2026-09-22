#!/usr/bin/env bash
# Put the blobs under test into Samples/.
#
# The probe needs two things: a metallib built for macOS (the thing whose
# acceptance on iOS is the open question) and a metallib built for iOS (a
# control, so that a refusal of the first can be told apart from the probe
# itself being broken).
#
# Preferred source is the Metal toolchain: compiling probe.metal twice gives
# two blobs that differ *only* in target, which is the cleanest possible pair.
# If the toolchain component is not installed, fall back to copying one of
# each off this machine — Apple ships both, and for a yes/no on the loader any
# genuine sample of each platform does.
#
# Samples/ is gitignored. Apple's compiled shaders are third-party bytes and
# do not belong in this repository.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p Samples
rm -f Samples/*.metallib Samples/*.mtlbsample

if xcrun -sdk macosx --find metal >/dev/null 2>&1 && xcrun -sdk macosx --find metallib >/dev/null 2>&1; then
    echo "Metal toolchain present — compiling probe.metal for both targets."
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    xcrun -sdk macosx   metal -c probe.metal -o "$tmp/mac.air"
    xcrun -sdk macosx   metallib "$tmp/mac.air" -o Samples/compiled-macos.metallib
    xcrun -sdk iphoneos metal -c probe.metal -o "$tmp/ios.air"
    xcrun -sdk iphoneos metallib "$tmp/ios.air" -o Samples/compiled-ios.metallib
else
    echo "Metal toolchain not installed (xcodebuild -downloadComponent MetalToolchain)."
    echo "Falling back to metallibs already on this machine."

    mac_src=/System/Library/CoreServices/default.metallib
    ios_src=$(/usr/bin/find /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs \
                 -name '*.metallib' 2>/dev/null | head -1)

    [ -r "$mac_src" ] || { echo "no macOS metallib found at $mac_src" >&2; exit 1; }
    cp "$mac_src" Samples/system-macos.metallib
    echo "  macOS sample: $mac_src"

    if [ -n "$ios_src" ]; then
        cp "$ios_src" Samples/system-ios.metallib
        echo "  iOS control:  $ios_src"
    else
        echo "  WARNING: no iOS metallib found; the control will be missing." >&2
    fi
fi

# Flatten fat archives and move off the .metallib extension, so sideloaders
# do not mistake a 0xCAFEBABE metallib for a universal Mach-O. See the README.
echo
./normalize-samples.py Samples

echo
ls -l Samples/
