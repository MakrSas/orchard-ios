# qemu-build.sh

Builds the vendored `vendor/qemu` submodule into `vendor/qemu/build/qemu-system-<arch>`, linking the
thin device shim(s) to **`crates/reims-vgpu`**.

| Target | Typical host | Output | Pathway |
|--------|--------------|--------|---------|
| `aarch64` | Darwin / Apple Silicon | `qemu-system-aarch64` | arm64 macOS guest on macOS host (`vm/boot-arm64.sh`) |
| `x86_64` | Linux | `qemu-system-x86_64` | x86 macOS guest on Linux host (`vm/boot-x86.sh`) |

| Backend | Where it runs | Role |
|---------|---------------|------|
| `metal` | Apple Silicon | Metal-direct encode for the arm64 macOS guest |
| `vulkan` | Linux native ICD or Apple Silicon MoltenVK | Vulkan encode + `metal2vulkan` for x86/Linux and arm64/MoltenVK pathways |

Defaults: target by host OS (`aarch64` on Darwin, `x86_64` on Linux); backend defaults by host OS
(`metal` on Darwin, `vulkan` elsewhere) — **override for the pathway you are on**.

## What it does

`vendor/qemu` already carries the project patches — this script does **not** clone or patch, it
builds. Steps:

1. Populates the submodule if needed (`git submodule update --init vendor/qemu`).
2. Resolves `--target` (or `QEMU_TARGET`) and `--backend metal|vulkan` (or `REIMS_VGPU_BACKEND`).
3. Builds `crates/reims-vgpu` as a staticlib and links it into the device shim:
   - **Apple + metal:** real MTL frameworks + encode path.
   - **Non-Apple + metal:** rejected; Metal is Apple-only.
   - **vulkan:** ash-based in-crate engine (native Vulkan on Linux, MoltenVK on macOS).
4. **aarch64:** expects `CONFIG_VMAPPLE`, HVF/Cocoa configure, verifies `-M vmapple`.
5. **x86_64:** `x86_64-softmmu`, HVF/Cocoa off; lists PCI/sysbus device help as applicable.

Re-runs are idempotent (skips configure when the target/backend stamp matches). Switching target
or backend forces reconfigure. Patch record: `vendor/qemu-patches/`.

## Run

```sh
# Explicit pathway builds
scripts/qemu-build/qemu-build.sh --target aarch64 --backend metal
scripts/qemu-build/qemu-build.sh --target aarch64 --backend vulkan
REIMS_VGPU_BACKEND=vulkan scripts/qemu-build/qemu-build.sh --target x86_64

# Point the matching boot script at the binary
QEMU_BIN=$PWD/vendor/qemu/build/qemu-system-aarch64 vm/boot-arm64.sh --device reims-vgpu-mmio --testing
QEMU_BIN=$PWD/vendor/qemu/build/qemu-system-x86_64 vm/boot-x86.sh --device reims-vgpu-pci --testing
```

### Requirements

- **Both:** cargo (`crates/reims-vgpu`), ninja, meson, pkg-config, glib, pixman.
- **aarch64 + metal:** macOS, Xcode CLT, HVF/Cocoa.
- **aarch64 + vulkan:** macOS, Xcode CLT, HVF/Cocoa, Vulkan loader, and MoltenVK ICD.
- **x86_64:** Linux QEMU build deps; KVM for boots.
