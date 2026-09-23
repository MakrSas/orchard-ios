#!/usr/bin/env python3
"""Enumerate the guest's CGL renderers and say which are hardware.

Runs inside the macOS guest. Browsers do not report *why* they are on a software
renderer, only that they are; this asks the OS directly, through
`CGLQueryRendererInfo`, which is the same enumeration OpenGL.framework uses to
pick a renderer. `ctypes` is enough — the guest has no compiler and installing
one would change the snapshot under measurement.

The property that matters is `kCGLRPAccelerated` (1636). A renderer list whose
only entry is the Apple Software Renderer is the whole explanation for
"hardware acceleration disabled" in every OpenGL-based browser at once.
"""

import ctypes
import ctypes.util
import sys

OGL = ctypes.CDLL("/System/Library/Frameworks/OpenGL.framework/OpenGL")

# CGLRendererProperty, from CGLTypes.h. These are the renderer-property
# enumerators (small integers) — NOT the CGLPixelFormatAttribute values in the
# 1600s, which is an easy and silent confusion: CGLDescribeRenderer just returns
# an error for every one of them and the whole report reads as "no properties",
# which looks like a broken GPU rather than a broken probe.
PROPS = {
    "rendererID": 70,
    "accelerated": 73,
    "backingStore": 76,
    "window": 80,
    "compliant": 83,
    "displayMask": 84,
    "gpuVertexProcessing": 122,
    "gpuFragmentProcessing": 123,
    "online": 129,
    "acceleratedCompute": 130,
    "videoMemoryMegabytes": 131,
    "textureMemoryMegabytes": 132,
    "majorGLVersion": 133,
}

# kCGLRendererGenericFloatID — the Apple Software Renderer, which is what a
# machine with no accelerated driver falls back to.
APPLE_SOFTWARE_RENDERER_ID = 0x00020400


def query():
    info = ctypes.c_void_p()
    count = ctypes.c_int(0)
    # display mask 0xffffffff = "any display"
    err = OGL.CGLQueryRendererInfo(ctypes.c_uint32(0xFFFFFFFF), ctypes.byref(info), ctypes.byref(count))
    if err != 0:
        print(f"CGLQueryRendererInfo failed: {err}", file=sys.stderr)
        return 1
    print(f"renderers: {count.value}")
    accelerated = 0
    for i in range(count.value):
        vals = {}
        for name, prop in PROPS.items():
            out = ctypes.c_int(0)
            if OGL.CGLDescribeRenderer(info, ctypes.c_int(i), ctypes.c_int(prop), ctypes.byref(out)) == 0:
                vals[name] = out.value
        rid = vals.get("rendererID", 0)
        soft = (rid & 0x00FFFF00) == (APPLE_SOFTWARE_RENDERER_ID & 0x00FFFF00)
        if vals.get("accelerated"):
            accelerated += 1
        print(
            f"  [{i}] rendererID=0x{rid:08x} accelerated={vals.get('accelerated')} "
            f"online={vals.get('online')} window={vals.get('window')} "
            f"vram_mb={vals.get('videoMemoryMegabytes')} "
            f"gl_major={vals.get('majorGLVersion')} "
            f"{'APPLE_SOFTWARE_RENDERER' if soft else ''}"
        )
    OGL.CGLDestroyRendererInfo(info)
    print(f"accelerated_renderers: {accelerated}")
    return 0


if __name__ == "__main__":
    sys.exit(query())
