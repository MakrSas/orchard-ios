#!/usr/bin/env bash
# Build UEFI GOP for reims-vgpu-pci and wrap as a PCI option ROM (EFI header).
# Source crate: crates/reims-vgpu-efi
# Output: crates/reims-vgpu-efi/out/reims-vgpu-gop.rom
# Same PCI device (0x106B:0xEEEE) — not a second display.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/reims-vgpu-efi-rom/ → crate root crates/reims-vgpu-efi/
EFI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$EFI_DIR/out"
VENDOR=0x106B
DEVICE=0xEEEE

mkdir -p "$OUT_DIR"
rustup target add x86_64-unknown-uefi >/dev/null 2>&1 || true

echo "reims-vgpu-efi-rom: building UEFI GOP (x86_64-unknown-uefi) ..."
(
  cd "$EFI_DIR"
  cargo build --release --target x86_64-unknown-uefi
)

EFI_BIN="$EFI_DIR/target/x86_64-unknown-uefi/release/reims-vgpu-efi.efi"
if [ ! -f "$EFI_BIN" ]; then
  EFI_BIN="$EFI_DIR/target/x86_64-unknown-uefi/release/reims-vgpu-efi"
fi
[ -f "$EFI_BIN" ] || {
  echo "reims-vgpu-efi-rom: missing built efi image" >&2
  exit 1
}

cp -f "$EFI_BIN" "$OUT_DIR/reims-vgpu-efi.efi"

python3 - "$OUT_DIR/reims-vgpu-efi.efi" "$OUT_DIR/reims-vgpu-gop.rom" "$VENDOR" "$DEVICE" <<'PY'
"""Wrap PE32+ EFI image as UEFI PCI Expansion ROM (EFI signature 0x0EF1 + PCIR).

Layout (UEFI 2.x EFI_PCI_EXPANSION_ROM_HEADER + PCI_DATA_STRUCTURE):
  [0x00] 55 AA, init size, EFI sig 0x0EF1, subsystem, machine, ...
  [0x16] EfiImageHeaderOffset
  [0x18] PcirOffset
  [pcir] PCIR + vendor/device + class display + code type EFI
  [img]  PE/COFF EFI image (padded to 512)

The PE OptionalHeader.Subsystem MUST be BOOT_SERVICE_DRIVER (11). An
efi_app (10) is unloaded when StartImage returns — protocols vanish and
OpenCore reports Missing GOP. We force subsystem 11 even if rustc/link
defaults to efi_app.
"""
import struct
import sys

efi_path, rom_path, vendor_s, device_s = sys.argv[1:5]
vendor = int(vendor_s, 0)
device = int(device_s, 0)
efi = bytearray(open(efi_path, "rb").read())
if efi[:2] != b"MZ":
    sys.exit(f"not a PE/EFI image: {efi_path}")

# Force PE subsystem = BOOT_SERVICE_DRIVER (11).
e_lfanew = struct.unpack_from("<I", efi, 0x3C)[0]
pe = e_lfanew
if efi[pe : pe + 4] != b"PE\0\0":
    sys.exit("PE signature missing")
opt_magic = struct.unpack_from("<H", efi, pe + 24)[0]
# PE32+ OptionalHeader.Subsystem at +68 from start of optional header.
if opt_magic != 0x20B:
    sys.exit(f"expected PE32+ (0x20B), got {opt_magic:#x}")
sub_off = pe + 24 + 68
old_sub = struct.unpack_from("<H", efi, sub_off)[0]
struct.pack_into("<H", efi, sub_off, 11)
print(f"PE subsystem {old_sub} -> 11 (BOOT_SERVICE_DRIVER)")

# Compact header like QEMU efi-*.rom EFI images: PCIR at 0x1C, image at 0x40.
PCIR_OFF = 0x1C
EFI_OFF = 0x40
assert EFI_OFF >= PCIR_OFF + 0x18

# EFI_PCI_EXPANSION_ROM_HEADER (IndustryStandard/pci.h)
hdr = bytearray(EFI_OFF)
struct.pack_into("<H", hdr, 0x00, 0xAA55)  # Signature (on disk: 55 AA)
# InitializationSize filled later (512-byte units of whole ROM)
struct.pack_into("<I", hdr, 0x04, 0x00000EF1)  # EfiSignature
struct.pack_into("<H", hdr, 0x08, 0x000B)  # EfiSubsystem: BOOT_SERVICE_DRIVER=11
struct.pack_into("<H", hdr, 0x0A, 0x8664)  # EfiMachineType: X64
struct.pack_into("<H", hdr, 0x0C, 0x0000)  # CompressionType: uncompressed
# 0x0E..0x15 reserved zeros
struct.pack_into("<H", hdr, 0x16, EFI_OFF)  # EfiImageHeaderOffset
struct.pack_into("<H", hdr, 0x18, PCIR_OFF)  # PcirOffset

# PCI_DATA_STRUCTURE (rev 0 classic; code type EFI) — length 0x18
pcir = bytearray(0x18)
pcir[0:4] = b"PCIR"
struct.pack_into("<H", pcir, 0x04, vendor)
struct.pack_into("<H", pcir, 0x06, device)
struct.pack_into("<H", pcir, 0x08, 0)  # reserved / device list
struct.pack_into("<H", pcir, 0x0A, 0x18)  # structure length
pcir[0x0C] = 0x00  # revision
# Class code 0x030000 = VGA-compatible display (matches device class)
pcir[0x0D] = 0x00  # PI
pcir[0x0E] = 0x00  # subclass
pcir[0x0F] = 0x03  # base class
# ImageLength (0x10) filled later
struct.pack_into("<H", pcir, 0x12, 0x0000)  # code revision
pcir[0x14] = 0x03  # CodeType EFI
pcir[0x15] = 0x80  # Indicator last image
struct.pack_into("<H", pcir, 0x16, 0x0000)  # reserved

hdr[PCIR_OFF : PCIR_OFF + len(pcir)] = pcir

body = bytes(hdr) + bytes(efi)
pad = (512 - (len(body) % 512)) % 512
rom = bytearray(body + bytes(pad))
blocks = len(rom) // 512
struct.pack_into("<H", rom, 0x02, blocks)  # InitializationSize
struct.pack_into("<H", rom, PCIR_OFF + 0x10, blocks)  # ImageLength

open(rom_path, "wb").write(rom)
print(
    f"wrote {rom_path} ({len(rom)} bytes, {blocks}×512, "
    f"vendor={vendor:#x} device={device:#x}, efi_off={EFI_OFF:#x})"
)
# Structural self-check (layout + PE subsystem)
assert rom[0] == 0x55 and rom[1] == 0xAA
assert struct.unpack_from("<I", rom, 4)[0] == 0x0EF1
assert rom[PCIR_OFF : PCIR_OFF + 4] == b"PCIR"
assert rom[PCIR_OFF + 0x14] == 0x03  # CodeType EFI
assert rom[EFI_OFF : EFI_OFF + 2] == b"MZ"
e_lfanew2 = struct.unpack_from("<I", rom, EFI_OFF + 0x3C)[0]
pe2 = EFI_OFF + e_lfanew2
assert struct.unpack_from("<H", rom, pe2 + 24 + 68)[0] == 11
print("layout ok: 55AA + EFI sig + PCIR + MZ + PE subsystem=11")
PY

echo "reims-vgpu-efi-rom: $OUT_DIR/reims-vgpu-gop.rom"
ls -la "$OUT_DIR/"
