#!/usr/bin/env python3
"""Flatten metallib samples to single plain MTLB blobs.

A fat metallib archive begins with 0xCAFEBABE, which is also Mach-O's
FAT_MAGIC. Sideloaders walk an app bundle looking for Mach-O binaries to sign,
hit that magic on a .metallib, and refuse the install ("Invalid mach-o file!").
The file is not code and not damaged — it just shares four bytes with code.

So each sample is reduced to one MTLB member and written with a .mtlbsample
extension, which nothing treats as a shader library or as a binary.
"""
import struct, sys, pathlib

MTLB = b'MTLB'
FAT = b'\xca\xfe\xba\xbe'
PLATFORM_OFFSET = 0x0B          # see README: 0x81 macOS, 0x82 iOS, 0x87 sim


def slices(data: bytes):
    if data[:4] == MTLB:
        return [data]
    if data[:4] != FAT:
        return []
    out = []
    (count,) = struct.unpack('>I', data[4:8])
    for i in range(count):
        e = 8 + i * 20
        if len(data) < e + 20:
            break
        off, size = struct.unpack('>II', data[e + 8:e + 16])
        if 0 <= off and size > 0 and off + size <= len(data):
            member = data[off:off + size]
            if member[:4] == MTLB:
                out.append(member)
    return out


def main(argv):
    if len(argv) != 2:
        print(__doc__); return 2
    d = pathlib.Path(argv[1])
    for src in sorted(d.glob('*.metallib')):
        parts = slices(src.read_bytes())
        if not parts:
            print(f"  !! {src.name}: no MTLB member found, skipped"); continue
        # Prefer the newest member; they differ by AIR language version and the
        # question being asked is about today's compiler, not a legacy slice.
        best = max(parts, key=lambda b: b[0x08])
        dst = src.with_suffix('.mtlbsample')
        dst.write_bytes(best)
        src.unlink()
        plat = {0x81: 'macOS', 0x82: 'iOS', 0x87: 'iOS-Sim'}.get(best[PLATFORM_OFFSET], 'unknown')
        note = f"(flattened from {len(parts)} members)" if len(parts) > 1 else ""
        print(f"  {dst.name}: {plat}, airLang={best[0x08]}, {len(best)} bytes {note}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
