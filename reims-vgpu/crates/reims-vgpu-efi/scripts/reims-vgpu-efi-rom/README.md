# reims-vgpu-efi-rom

Build the **UEFI Graphics Output Protocol** PE for product PCI `reims-vgpu-pci`
and wrap it as a PCI expansion ROM. Lives next to the crate it packages
(`crates/reims-vgpu-efi/`).

This is **not** a second display or `-vga std`. GOP is published by a PCI
**option ROM** that OVMF loads when it enumerates our VGA-class function. The
host linear framebuffer is **BAR1 on the same device** (see
`vendor/qemu/hw/display/reims-vgpu-pci.c`).

## Build

```sh
crates/reims-vgpu-efi/scripts/reims-vgpu-efi-rom/reims-vgpu-efi-rom.sh
# → crates/reims-vgpu-efi/out/reims-vgpu-gop.rom
# → crates/reims-vgpu-efi/out/reims-vgpu-efi.efi
```

Requires Rust `x86_64-unknown-uefi` target (`rustup target add x86_64-unknown-uefi`).

## Boot

`vm/boot-x86.sh --device reims-vgpu-pci` auto-attaches the ROM when present:

```
-device reims-vgpu-pci,...,romfile=.../reims-vgpu-gop.rom,rombar=1
```

- `REIMS_VGPU_GOP_ROM=/path/to.rom` — override path
- `REIMS_VGPU_GOP_ROM=` — disable option ROM

## Packaging invariants

The packager asserts:

1. `55 AA` + EFI signature `0x0EF1` + `PCIR` code type `0x03` + `MZ` PE image  
2. **PE `OptionalHeader.Subsystem = 11` (BOOT_SERVICE_DRIVER)** — `efi_app` (10)
   is unloaded when `StartImage` returns, so OpenCore never sees a stable GOP  

Success line on serial (stable scrape tag):

```
reims-vgpu efi-gop: GOP installed
```

See the crate README (`crates/reims-vgpu-efi/README.md`) for source layout and
product context.
