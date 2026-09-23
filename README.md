# Orchard for iPhone

**Apple Silicon macOS, running inside an iPhone app.**

An arm64 macOS Ventura guest boots through Apple's own chain — `AVPBooter →
iBoot → XNU` — on QEMU's `apple-vm` machine, and runs to the desktop. QEMU is
a library inside the app, the guest's cores are translated by TCG with a JIT,
and the GPU is [reims-vgpu](https://github.com/steelbrain/reims-vgpu), drawing
through the phone's own Metal. It is slow — around 15 frames a second at best
on an iPhone 15 — but it is the real macOS.

This is the iOS port of [Orchard](https://github.com/yaelliethy/orchard), which
does the same on a Linux host; that guide is in [`LINUX.md`](LINUX.md).

Nothing Apple ships is in the app or this repository. You bring the macOS VM.

## What you need

| | |
|---|---|
| Phone | Tested on iPhone 15 (A16, 6 GB) with iOS 27. iOS 16 or later; the more memory, the better — a phone with 6 GB or more. |
| Sideloading | A signing tool that adds the **increased memory limit** capability to the app's ID when it signs it. Stock iLoader does not; the build used in testing was patched to. Without it iOS kills the app at about 3 GB; with it the ceiling is about 4 GB. |
| JIT | A way to attach a debugger to the app on launch, such as StikDebug. Without JIT the app cannot start the machine. |
| Mac | Only to make the VM: a Mac with [UTM](https://mac.getutm.app) and a macOS 13 (Ventura) VM made with the **Virtualize** backend. |
| Space | About 20 GB on the phone or on a USB drive for the VM folder. |

## 1. Make the VM folder (on a Mac)

Install macOS Ventura in UTM with **Virtualize**, finish its setup assistant
there (it is far too slow to do on the phone), then shut it down — not
suspend — and convert it. The converter needs `qemu-img`, from Homebrew's
QEMU:

```bash
brew install qemu
```

```bash
python3 scripts/utm-to-orchard.py ~/Library/Containers/com.utmapp.UTM/Data/Documents/Ventura.utm --out-dir ~/Desktop/OrchardVM
```

The folder holds `disk.qcow2`, `aux.img`, `config.json`,
`AVPBooter.patched.bin` and `overlay.qcow2`. The disk is only ever read; all
the guest's writes go into `overlay.qcow2`. To reset the guest, replace the
overlay with a fresh one:

```bash
qemu-img create -f qcow2 -F qcow2 -b disk.qcow2 -u overlay.qcow2 68719476736
```

## 2. Install the app

Take `Orchard.ipa` from the release (or build it, below) and install it with
your signing tool, with the increased memory limit on. Then enable JIT for it
with StikDebug and launch it.

## 3. Give it the VM

Either copy the `OrchardVM` folder into **Files → On My iPhone → Orchard**,
or open the round button in the corner → **Settings… → Where the VM lives →
Choose a folder…** and pick it on a USB drive. From a drive the VM boots in
place; a USB SSD is noticeably faster than a flash stick.

## 4. Settings that matter

In the round button's **Settings…**. All of these apply the next time the
machine starts.

* **Machine → Memory**: 2.5 GB with the increased limit, 2 GB without it.
  Under 2 GB macOS reboots in a loop; at 3 GB the app hits the iOS ceiling as
  soon as a few windows open. The ceiling as measured is shown under the
  picker.
* **Machine → Cores**: 2–4. An iPhone has two fast cores; more guest cores
  than that are not always faster.
* **Translator → Translation buffer**: 256 MB. Less makes the guest
  re-translate its code constantly; more can hang the machine at the first
  instruction.
* **Screen → Resolution**: the "Like the screen" modes fill the phone edge
  to edge; smaller is faster. **Refresh rate**: 30 Hz is a quarter of the
  work of 120.
* **Screen → Trackpad mode**: the pointer moves by how far the finger travels;
  tap to click, two fingers to right-click and scroll, hold then move to drag.

## 5. Start it

Tap the round button in the corner and choose **Start**. The first boot to
the login window takes several minutes. A machine that has stopped cannot be
started again in the same run: close the app and open it again.

## Sharing files with the guest

Put files in **Files → On My iPhone → Orchard → Shared**, and in the guest's
Finder choose **Go → Connect to Server** (⌘K), enter `http://10.0.2.2:8080`,
**Connect**, then **Registered User** with any name and password — nothing
checks them, but Ventura's WebDAV client will not mount without a user name,
so **Guest** fails. The folder is mounted like a network drive and works both
ways, around 15 MB/s. From the guest's Terminal, `curl` works as well:

```bash
curl -O http://10.0.2.2:8080/name-of-the-file
```

It can be turned off in **Settings… → Network**.

## Building from source

On a Mac with Xcode (and its iPhoneOS SDK), [rustup](https://rustup.rs) and
Homebrew. QEMU's configure runs once natively before the iOS build, which is
what the Homebrew libraries are for:

```bash
brew install ninja pkgconf glib pixman libslirp qemu
```

Then:

```bash
scripts/fetch-ios-deps.sh
```

```bash
scripts/build-ios.sh
```

```bash
ios/app/build.sh
```

The first downloads prebuilt static C libraries (GLib, pixman, libslirp and
others; see `scripts/ios-deps-SOURCES.md`) into `deps/ios`. The second builds
QEMU and reims-vgpu into `qemu/build-ios/libqemu-aarch64-softmmu.dylib`. The
third builds the app around it and writes `ios/Orchard.ipa`, unsigned — your
signing tool signs it on install.

`ios/PORTING.md` is the long account of the port (in Russian): what was
changed and why, and what each problem cost to find.

## Known problems

* It is slow. Every guest instruction is translated on the phone's CPU.
* Apps take a long time to open their first window; wait before tapping again.
* Internet works in the guest (virtio-net with NAT through the phone's own
  connection), but nothing outside the phone can connect in to it.
* Sound is experimental: turn it on in **Settings… → Machine → Sound**. The
  guest sees a virtio-sound card as "Speakers"; while its CPUs are busy the
  sound can break up.
* If the app closes by itself a while after starting, iOS took the memory:
  lower the guest's memory or the translation buffer.

## What is in here

| Path | What it is |
|---|---|
| `ios/app/` | The iPhone app: SwiftUI, no Xcode project; `ios/app/build.sh` builds it. |
| `qemu/` | QEMU with Orchard's `apple-vm` machine and the iOS changes: runs as a library, JIT on iOS, in-process display and input. |
| `reims-vgpu/` | The paravirtual GPU, vendored — [steelbrain/reims-vgpu](https://github.com/steelbrain/reims-vgpu), with a Metal backend that builds for iOS. |
| `scripts/` | Build, convert a UTM VM, fetch the iOS dependencies. |
| `ios/PORTING.md` | How the port was done, and what it cost. |
| `TECHNICAL.md` | How Orchard itself works. |

## Credits

- **Youssef Elliethy** ([yaelliethy](https://github.com/yaelliethy)) — Orchard:
  reverse-engineered enough of Apple's bootloader, virtualization and OS stack
  to boot macOS on QEMU, and ported reims-vgpu to it.
- **Anees Iqbal** ([steelbrain](https://github.com/steelbrain)) — reims-vgpu,
  the GPU everything on screen goes through.
- **Visual Ehrmanntraut** and **ChefKiss** ([ChefKissInc](https://github.com/ChefKissInc))
  — Inferno, whose work laid the groundwork, including the Apple pointer
  authentication macOS cannot boot without.
- **Alexander Graf** — QEMU's `vmapple` machine.
- **NyanSatan** — [Virtual-iBoot-Fun](https://github.com/NyanSatan/Virtual-iBoot-Fun),
  the AVPBooter patch.
- **Makr** ([MakrSas](https://github.com/MakrSas)), with Claude — the iOS port.

`ATTRIBUTION.md` says exactly what came from where.

## Licence

QEMU and the changes to it, and the iOS app, are GPL-2.0-or-later; reims-vgpu
is LGPL-3.0, with its author's permission for use in this app. The details,
and the licences of the bundled C libraries, are in `LICENSE-NOTICE.md`.

Unofficial, and not affiliated with Apple or ChefKiss. Apple's licence permits
macOS virtualisation only on Apple-branded hardware — check that your use is
within it.
