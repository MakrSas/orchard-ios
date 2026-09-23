# Attribution

This tree combines several bodies of work.

## QEMU

`qemu/` is a checkout of [upstream QEMU](https://github.com/qemu/qemu) with the
changes listed in `TECHNICAL.md` applied. QEMU is licensed GPL-2.0-only with
parts under compatible licences; see `qemu/LICENSE`. Our changes are
GPL-2.0-or-later.

The upstream `vmapple` machine (`hw/vmapple/`) is by Alexander Graf and
contributors; this tree extends it.

## Inferno

Parts of the Apple pointer-authentication support in `target/arm` are derived
from [Inferno](https://github.com/ChefKissInc/Inferno), ChefKissInc's fork of
QEMU (GPL-2.0-or-later):

* `target/arm/helper.c` — the `KERNELKEYLO/HI_EL1`, `APCTL_EL1` and `APCFG_EL1`
  register definitions, the key-write diversifier and `apctl_write`.
* `target/arm/tcg/pauth_helper.c` — mixing the kernel key into EL1 signatures,
  and enabling the keys in Apple mode.
* `target/arm/cpu.h` — the fields and bit definitions those need.

Everything else in `target/arm` here (the per-core key snapshot, PSCI CPU_ON key
inheritance, the WFE park) is ours.

## reims-vgpu

`reims-vgpu/` is **[steelbrain/reims-vgpu](https://github.com/steelbrain/reims-vgpu)**,
not our work. It is the paravirtual GPU: Apple's ParavirtualizedGraphics
protocol, AIR shader translation (via the `metal2vulkan` crate) and a Vulkan
renderer. Licensed **LGPL-3.0** (see `reims-vgpu/LICENSE`). This tree includes
it and links it into QEMU as a static library through the C ABI in
`reims-vgpu/crates/reims-vgpu/include/reims_vgpu_qemu_abi.h`; the QEMU-side
transports (`hw/display/reims-vgpu-*.c`) are ours.

## Virtual-iBoot-Fun

The AVPBooter patch — which routine has to be stubbed for Apple's VM firmware to
run outside Apple's own hypervisor — is from NyanSatan's
[Virtual-iBoot-Fun](https://github.com/NyanSatan/Virtual-iBoot-Fun). Ours is
only `scripts/patch-avpbooter.py`, which applies it to a firmware image the user
supplies after checking that the bytes at the site are what the patch expects.

## The iOS port

`ios/app` and the iOS-specific parts of `qemu/` (the in-process display and
input in `ui/orchard-embed.c`, the JIT and coroutine changes that let QEMU run
as a library inside an iOS app) come from Makr's
[Inferno-iOS](https://github.com/MakrSas/Inferno-iOS), the iPhone port of
ChefKissInc's Inferno, and were carried over and extended here with Claude.
Here they are GPL-2.0-or-later, relicensed by their author from Inferno-iOS's
GPL-3.0 (the app) and AGPL-3.0 (`ui/inferno-embed.c`).

The iOS build links a set of static C libraries (GLib, pixman, libslirp, GMP,
Nettle and others) built from unmodified upstream sources; their versions and
licences are listed in `scripts/ios-deps-SOURCES.md` and shipped inside the
archive `scripts/fetch-ios-deps.sh` downloads.

## Not distributed here

* **AVPBooter** — Apple firmware. Extracted at setup time from the macOS image
  you download, where `Virtualization.framework` carries it.
* **macOS** — fetched at run time from the public `cirruslabs/macos-ventura-base`
  image. Apple's licence permits macOS virtualisation only on Apple-branded
  hardware; check your use against it.
