# macOS on QEMU, on a Linux host

> **On an iPhone:** the same machine runs inside an iOS app — see
> [`ios/README.md`](ios/README.md).

Video link: https://jumpshare.com/share/xWiBdnqUZPoGVcCptEGG

Boots an **arm64 macOS guest** (Ventura) on **QEMU/TCG on Linux**, through the
genuine Apple boot chain — `AVPBooter → iBootStage1 (LLB) → iBootStage2 (iBoot)
→ stock XNU` — with a GPU that renders the desktop on the host's Vulkan.

No Apple hardware is involved at run time, and nothing Apple ships is
redistributed here: you supply the firmware and the disk image, and the scripts
below fetch or patch them in place.

```
                                  guest
   ┌──────────────────────────────────────────────────────────────┐
   │ macOS 13 (arm64e, RELEASE_ARM64_VMAPPLE) — unmodified        │
   └──────────────────────────────────────────────────────────────┘
        │ PVG command stream            │ virtio-blk (aux + root)
   ┌────┴──────────────┐          ┌─────┴───────────────────────┐
   │ reims-vgpu        │          │ QEMU `apple-vm` machine     │
   │ (Rust, Vulkan)    │          │ vmapple devices + AVPBooter │
   └────┬──────────────┘          └─────────────────────────────┘
        │ Vulkan
   ┌────┴──────────────┐
   │ host GPU (RADV)   │
   └───────────────────┘
```

## What you need

| | |
|---|---|
| Host | Linux, x86-64. A Vulkan GPU (developed on AMD RENOIR / Mesa RADV). |
| RAM | 16 GB+ — the guest gets 10 GB by default, and the whole RAMBlock is imported into one Vulkan heap, so the host needs that much importable GPU memory too. |
| Disk | ~40 GB for the macOS image (50 GB apparent, sparse). |
| Tools | `cargo` (stable Rust), a C toolchain, `ninja`, `meson`, `python3` with `requests` and `lz4`. |
| Extra tool | [`apfs-fuse`](https://github.com/sgan81/apfs-fuse), to read the macOS image — that is where the firmware comes from. |

## Getting it running

```sh
# 1. build QEMU (this also builds the Rust GPU device)
scripts/build.sh

# 2. fetch the macOS image: disk, NVRAM, and the VM config that carries its ECID
scripts/fetch-tart-image.py --out-dir images        # ~30 GB, resumable

# 3. take Apple's VM firmware out of the image, and patch it to run here
scripts/extract-avpbooter.py images/disk.raw -o images/AVPBooter.vmapple2.bin
scripts/patch-avpbooter.py images/AVPBooter.vmapple2.bin images/AVPBooter.patched.bin

# 4. optional: ask the kernel for a serial log
scripts/prepare-aux.py images/aux.img --set 'boot-args=-v serial=3'

# 5. boot
scripts/boot-robust.sh      # or scripts/run-vm.sh to stay in the foreground
```

The desktop appears in the device's own Vulkan window. First boot to the login
screen takes a few minutes under TCG; see **Known problems** before concluding
anything has gone wrong.

## What is in here

| Path | What it is |
|---|---|
| `qemu/` | Upstream QEMU with our changes applied (see `TECHNICAL.md` for the list). |
| `reims-vgpu/` | The paravirtual GPU, vendored in-tree — [steelbrain/reims-vgpu](https://github.com/steelbrain/reims-vgpu), a Rust crate that turns Apple's PVG command stream into Vulkan, with our changes baked in. Not ours. |
| `scripts/` | Fetch, patch, build and run. |
| `TECHNICAL.md` | How the whole thing works, and what each change is for. |
| `CONTRIBUTING.md` | Where a change belongs, how to test it, and the evidence rule. |

## Supporting the project

[**buymeacoffee.com/yaelliethy**](https://buymeacoffee.com/yaelliethy)

Contributions go to hardware and to keeping this maintained. The first target is
an **arm64 laptop — a Snapdragon X2 Elite Extreme machine, about $1,600**.

That is not a wish-list item. Everything here runs the guest under TCG, because
the development host is x86-64 and the guest is arm64 — every boot, every test,
every bisect pays emulation cost. An arm64 host makes the KVM path **testable**
(see the KVM section in `TECHNICAL.md`, which is written as an open question
precisely because nobody has been able to try it), and should make development
considerably faster.

## Licensing and what is not distributed

The QEMU changes are GPL-2.0-or-later, like QEMU itself. `reims-vgpu/` is
[steelbrain/reims-vgpu](https://github.com/steelbrain/reims-vgpu), LGPL-3.0. Parts of the Apple PAC
support in `target/arm` are derived from the [Inferno](https://github.com/ChefKissInc/Inferno)
fork of QEMU (ChefKissInc, GPL-2.0-or-later); `TECHNICAL.md` says which.

The AVPBooter patch applied by `scripts/patch-avpbooter.py` is from NyanSatan's
[Virtual-iBoot-Fun](https://github.com/NyanSatan/Virtual-iBoot-Fun).

**Not included, and not redistributable:** nothing Apple ships is in this
repository. The macOS image is fetched at run time from the public
`cirruslabs/macos-ventura-base` image, and Apple's VM firmware
(`AVPBooter.vmapple2.bin`) is taken out of that image, where macOS itself
carries it in `Virtualization.framework`. Apple's licence permits macOS
virtualisation only on Apple-branded hardware — check that your use is within
it.

## A host quirk worth knowing

If QEMU's `configure` dies with

```
TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'
```

an old `packaging` in `~/.local/lib/python3.x/site-packages` is shadowing the
one in QEMU's build venv. `scripts/build.sh` already exports
`PYTHONNOUSERSITE=1`, which is the whole fix.
