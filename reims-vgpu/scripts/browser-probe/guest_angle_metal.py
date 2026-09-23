#!/usr/bin/env python3
"""Make ANGLE say, in its own words, why its Metal display will not initialize.

Run inside the guest:  python3 guest_angle_metal.py

WHY THIS EXISTS

With `--use-angle=metal`, Chrome's GPU process dies reporting only

    Initialization of all (1) EGL display types failed.
    GLDisplayEGL::Initialize failed.

That message is Chrome counting failures, not ANGLE explaining one. On the
OpenGL backend ANGLE's own diagnostics reach the Chrome log through
`angle_platform_impl`; on the Metal backend **nothing** is printed, because the
failure happens inside `eglInitialize` before that channel carries anything. So
the one line that would name the cause is exactly the line Chrome never gets.

ANGLE implements `EGL_KHR_debug`, which delivers those messages to a callback.
This driver loads Chrome's own `libEGL.dylib`, installs that callback, asks for
a Metal display explicitly, and prints whatever ANGLE says — with no browser,
no sandbox and no GPU process in the way. The OpenGL backend is initialized
afterwards as a control: if Metal fails and OpenGL succeeds in the same process,
the failure is specific to the Metal path rather than to this harness.
"""
import ctypes
import sys

CHROME_LIBS = ("/Applications/Google Chrome.app/Contents/Frameworks/"
               "Google Chrome Framework.framework/Versions/Current/Libraries/")

# A bare ctypes process has no CoreGraphics, and Metal's system-default-device
# lookup then latches nil for the life of the process behind a dispatch_once.
# ANGLE asks for that device, so without this line the probe would manufacture
# the very failure it is trying to diagnose. See guest_metal_caps.py.
ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")

# EGL constants (from the EGL and EGL_ANGLE_platform_angle headers).
EGL_NO_DISPLAY = 0
EGL_PLATFORM_ANGLE_ANGLE = 0x3202
EGL_PLATFORM_ANGLE_TYPE_ANGLE = 0x3203
EGL_PLATFORM_ANGLE_TYPE_OPENGL_ANGLE = 0x320D
EGL_PLATFORM_ANGLE_TYPE_METAL_ANGLE = 0x3489
EGL_NONE = 0x3038
EGL_DEBUG_MSG_CRITICAL_KHR = 0x33B9
EGL_DEBUG_MSG_ERROR_KHR = 0x33BA
EGL_DEBUG_MSG_WARN_KHR = 0x33BB
EGL_DEBUG_MSG_INFO_KHR = 0x33BC
EGL_TRUE = 1
EGL_VENDOR = 0x3053
EGL_VERSION = 0x3054

EGL_ERROR_NAMES = {
    0x3000: "EGL_SUCCESS", 0x3001: "EGL_NOT_INITIALIZED",
    0x3002: "EGL_BAD_ACCESS", 0x3003: "EGL_BAD_ALLOC",
    0x3004: "EGL_BAD_ATTRIBUTE", 0x3005: "EGL_BAD_CONFIG",
    0x3006: "EGL_BAD_CONTEXT", 0x3007: "EGL_BAD_CURRENT_SURFACE",
    0x3008: "EGL_BAD_DISPLAY", 0x3009: "EGL_BAD_MATCH",
    0x300A: "EGL_BAD_NATIVE_PIXMAP", 0x300B: "EGL_BAD_NATIVE_WINDOW",
    0x300C: "EGL_BAD_PARAMETER", 0x300D: "EGL_BAD_SURFACE",
    0x300E: "EGL_CONTEXT_LOST",
}

DEBUGPROC = ctypes.CFUNCTYPE(
    None, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)


def load_egl():
    # libGLESv2 first: libEGL depends on it, and loading it by absolute path
    # avoids relying on @loader_path resolution from a foreign process.
    ctypes.CDLL(CHROME_LIBS + "libGLESv2.dylib", mode=ctypes.RTLD_GLOBAL)
    return ctypes.CDLL(CHROME_LIBS + "libEGL.dylib", mode=ctypes.RTLD_GLOBAL)


def main():
    egl = load_egl()
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglGetError.restype = ctypes.c_int32
    egl.eglInitialize.restype = ctypes.c_uint32
    egl.eglInitialize.argtypes = [ctypes.c_void_p,
                                  ctypes.POINTER(ctypes.c_int32),
                                  ctypes.POINTER(ctypes.c_int32)]
    egl.eglQueryString.restype = ctypes.c_char_p
    egl.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int32]

    messages = []

    @DEBUGPROC
    def on_message(error, command, msg_type, _thread, _object, message):
        text = (message or b"").decode(errors="replace").strip()
        cmd = (command or b"").decode(errors="replace")
        messages.append(f"    ANGLE[{EGL_ERROR_NAMES.get(error, hex(error))}] {cmd}: {text}")

    ctl = egl.eglGetProcAddress(b"eglDebugMessageControlKHR")
    if ctl:
        # `eglDebugMessageControlKHR` takes `const EGLAttrib *`, and EGLAttrib is
        # `intptr_t` — 64-bit here. `eglGetPlatformDisplayEXT` below takes the
        # EXT spelling, `const EGLint *`, which is 32-bit. Using one width for
        # both makes ANGLE walk off the end of the array and segfault the
        # process, which reads as "the Metal backend crashed" rather than "the
        # probe passed the wrong type".
        fn = ctypes.CFUNCTYPE(ctypes.c_int32, DEBUGPROC,
                              ctypes.POINTER(ctypes.c_ssize_t))(ctl)
        attrs = (ctypes.c_ssize_t * 9)(
            EGL_DEBUG_MSG_CRITICAL_KHR, EGL_TRUE,
            EGL_DEBUG_MSG_ERROR_KHR, EGL_TRUE,
            EGL_DEBUG_MSG_WARN_KHR, EGL_TRUE,
            EGL_DEBUG_MSG_INFO_KHR, EGL_TRUE,
            EGL_NONE)
        fn(on_message, attrs)
        print("EGL_KHR_debug callback: installed")
    else:
        print("EGL_KHR_debug callback: UNAVAILABLE (messages will be absent, "
              "so an empty reason below means unreported, not none)")

    get_display = egl.eglGetProcAddress(b"eglGetPlatformDisplayEXT")
    if not get_display:
        print("eglGetPlatformDisplayEXT missing — cannot select a backend")
        return 2
    get_display_fn = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32))(get_display)

    failures = 0
    for label, backend in (("metal", EGL_PLATFORM_ANGLE_TYPE_METAL_ANGLE),
                           ("opengl", EGL_PLATFORM_ANGLE_TYPE_OPENGL_ANGLE)):
        messages.clear()
        print(f"\n=== ANGLE backend: {label} ===")
        attrs = (ctypes.c_int32 * 3)(EGL_PLATFORM_ANGLE_TYPE_ANGLE, backend, EGL_NONE)
        dpy = get_display_fn(EGL_PLATFORM_ANGLE_ANGLE, None, attrs)
        if not dpy:
            err = egl.eglGetError()
            print(f"  eglGetPlatformDisplayEXT -> EGL_NO_DISPLAY "
                  f"({EGL_ERROR_NAMES.get(err, hex(err))})")
            print("\n".join(messages) or "    (ANGLE said nothing)")
            failures += 1
            continue
        major, minor = ctypes.c_int32(), ctypes.c_int32()
        ok = egl.eglInitialize(dpy, ctypes.byref(major), ctypes.byref(minor))
        if ok != EGL_TRUE:
            err = egl.eglGetError()
            print(f"  eglInitialize -> FAILED ({EGL_ERROR_NAMES.get(err, hex(err))})")
            print("\n".join(messages) or "    (ANGLE said nothing)")
            failures += 1
            continue
        vendor = egl.eglQueryString(dpy, EGL_VENDOR)
        version = egl.eglQueryString(dpy, EGL_VERSION)
        print(f"  eglInitialize -> OK, EGL {major.value}.{minor.value}")
        print(f"  vendor : {(vendor or b'').decode(errors='replace')}")
        print(f"  version: {(version or b'').decode(errors='replace')}")
        if messages:
            print("\n".join(messages))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
