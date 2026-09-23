#!/usr/bin/env python3
"""Ask the guest why `MTLCreateSystemDefaultDevice()` answers the way it does.

Run inside the guest:  python3 guest_default_device.py

WHY THIS EXISTS

A nil system default device is the whole reason Chrome has no GPU here:
ANGLE-Metal asks for the system default at display init and gives up when it is
nil, so the GPU process never boots. `guest_metal_caps.py` reports *that* it is
nil. It cannot report *why*, and the why decides whether this is something the
host can influence at all.

Reading Metal.framework's own selection path gives four exits, and they have
completely different consequences:

  1. the CoreGraphics dlsym fails         -> returns nil with NO fallback
  2. a `deviceNameMatch` preference is set and matches nothing -> nil
  3. the device array is empty            -> nil, and the device is really absent
  4. otherwise CGSCreateDefaultMetalDevice, and if that is nil, the first
     entry of the same array `MTLCopyAllDevices()` returns

Exit 4 is the important one: it means a nil default device alongside a non-empty
`MTLCopyAllDevices()` should be **impossible**. If this probe reports exactly
that, then the selection path is not what is being executed and the premise
needs re-deriving; if it reports a match on 1 or 2, the cause is guest process
environment and no host change can address it.

Each step is printed unconditionally and in order, so the output distinguishes
the exits by itself rather than needing a second run to narrow down.
"""
import ctypes
import os
import sys

libc = ctypes.CDLL(None)
libc.dlopen.restype = ctypes.c_void_p
libc.dlopen.argtypes = [ctypes.c_char_p, ctypes.c_int]
libc.dlsym.restype = ctypes.c_void_p
libc.dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
objc.sel_registerName.restype = ctypes.c_void_p
objc.sel_registerName.argtypes = [ctypes.c_char_p]

CF = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
CF.CFStringCreateWithCString.restype = ctypes.c_void_p
CF.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
CF.CFPreferencesCopyAppValue.restype = ctypes.c_void_p
CF.CFPreferencesCopyAppValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
CF.CFStringGetCString.restype = ctypes.c_bool
CF.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
CF.CFGetTypeID.restype = ctypes.c_ulong
CF.CFGetTypeID.argtypes = [ctypes.c_void_p]
CF.CFStringGetTypeID.restype = ctypes.c_ulong
kCFStringEncodingUTF8 = 0x08000100

RTLD_LAZY = 0x1
CORE_GRAPHICS = b"/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"


def cfstr(text):
    return CF.CFStringCreateWithCString(None, text.encode(), kCFStringEncodingUTF8)


def from_cfstr(ref):
    """Render a CFStringRef, or describe a non-string so a wrong type is visible."""
    if not ref:
        return None
    if CF.CFGetTypeID(ref) != CF.CFStringGetTypeID():
        return "<non-string CFTypeRef>"
    buf = ctypes.create_string_buffer(512)
    if CF.CFStringGetCString(ref, buf, len(buf), kCFStringEncodingUTF8):
        return buf.value.decode(errors="replace")
    return "<undecodable>"


def msg(restype, obj, selector, *args, argtypes=None):
    """One `objc_msgSend` call, with restype/argtypes bound explicitly.

    ctypes caches attributes on the shared function object, so a type left over
    from a previous call would silently apply to this one; rebinding per call is
    the price of not writing a compiled helper. argtypes is never derived from
    the values, because a derived type cannot express an out-parameter.
    """
    fn = objc.objc_msgSend
    fn.restype = restype
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + list(argtypes or [])
    return fn(obj, objc.sel_registerName(selector), *args)


def main():
    metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    metal.MTLCopyAllDevices.restype = ctypes.c_void_p

    # Step 3 first: the device array is what every other exit is judged against.
    arr = metal.MTLCopyAllDevices()
    count = msg(ctypes.c_ulonglong, arr, b"count") if arr else 0
    print(f"MTLCopyAllDevices count     : {count}")
    for i in range(count):
        dev = msg(ctypes.c_void_p, arr, b"objectAtIndex:", ctypes.c_ulonglong(i),
                  argtypes=[ctypes.c_ulonglong])
        name = msg(ctypes.c_void_p, dev, b"name")
        print(f"  [{i}] name                 : {from_cfstr(name)}")

    # Step 2: a name filter that matches nothing returns nil with a live device.
    pref = CF.CFPreferencesCopyAppValue(cfstr("deviceNameMatch"), cfstr("com.apple.Metal"))
    print(f"deviceNameMatch pref        : {from_cfstr(pref)!r}")

    # Anything that restricts registration by registry id, read at Metal init.
    for var in ("ALLOWED_GPU_IDS", "MTL_DEVICE_WRAPPER_TYPE", "MTL_HUD_ENABLED",
                "MTL_DEBUG_LAYER", "CGL_DEVICE_MASK"):
        if var in os.environ:
            print(f"env {var:24}: {os.environ[var]!r}")

    # Step 1: the one exit that returns nil with no fallback at all.
    handle = libc.dlopen(CORE_GRAPHICS, RTLD_LAZY)
    print(f"dlopen(CoreGraphics)        : {'ok' if handle else 'FAILED'}")
    sym = libc.dlsym(handle, b"CGSCreateDefaultMetalDevice") if handle else None
    print(f"dlsym(CGSCreateDefaultMetal): {'ok' if sym else 'NULL -> nil with no fallback'}")

    if sym:
        cgs = ctypes.CFUNCTYPE(ctypes.c_void_p)(sym)
        got = cgs()
        print(f"CGSCreateDefaultMetalDevice : {'nil' if not got else from_cfstr(msg(ctypes.c_void_p, got, b'name'))}")

    # And the question itself, asked last so every input above is already shown.
    default = metal.MTLCreateSystemDefaultDevice()
    print(f"MTLCreateSystemDefaultDevice: "
          f"{'nil' if not default else from_cfstr(msg(ctypes.c_void_p, default, b'name'))}")

    if not default and count:
        print("VERDICT: nil default device with a NON-EMPTY device array — the "
              "documented fallback did not run; re-derive the selection path.")
    elif not default:
        print("VERDICT: the device array is empty; the device is genuinely absent "
              "to Metal, so this is not a default-device selection problem.")
    else:
        print("VERDICT: the system default device resolves. Any browser that "
              "reports no GPU is failing for some other reason.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
