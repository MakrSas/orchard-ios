# Orchard on iPhone

The same `apple-vm` machine and reims-vgpu GPU as the Linux build, compiled
into an iOS app: QEMU runs as a library inside the app, the guest's cores are
translated by TCG with a JIT, and the GPU draws through the phone's own Metal.
An arm64 macOS Ventura guest boots to the desktop and runs, slowly — around
15 frames a second at best on an iPhone 15.

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
suspend — and convert it:

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
or open **Settings → Where the VM lives → Choose a folder…** and pick it on a
USB drive. From a drive the VM boots in place; a USB SSD is noticeably faster
than a flash stick.

## 4. Settings that matter

All of these apply the next time the machine starts.

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

The first boot to the login window takes several minutes.

## Building from source

On a Mac with Xcode (and its iPhoneOS SDK), rustup, ninja, pkg-config and
Python 3:

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

`ios/PORTING.md` is the long account of the port: what was changed and why,
and what each problem cost to find.

## Known problems

* It is slow. Every guest instruction is translated on the phone's CPU.
* Apps take a long time to open their first window; wait before tapping again.
* No sound yet, and no way to reach the guest from outside the phone.
* If the app closes by itself a while after starting, iOS took the memory:
  lower the guest's memory or the translation buffer.
