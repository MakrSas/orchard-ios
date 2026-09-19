# How this works

Three things have to be true at once for a stock macOS arm64 kernel to run on a
Linux host: the **boot chain** must complete, the **CPU** must behave like an
Apple virtual machine, and the **GPU** must answer Apple's paravirtual graphics
protocol. This document takes them in that order, then lists what we changed and
what is still broken.

---

## 1. The boot chain

Apple's VM firmware chain is unmodified except for one instruction pair:

```
AVPBooter (-bios)  →  iBootStage1 (LLB)  →  iBootStage2 (iBoot)  →  XNU
```

Each stage verifies the next, so nothing downstream of AVPBooter can be patched
without breaking image4 verification — and none of it is. What makes the chain
work is supplying the three things it expects to find:

**The real installed disk.** The tart image has the layout `fsboot` requires:
GPT part0 `iBootSystemContainer` (0.5 GB, APFS) + part1 `Container` (41 GB,
APFS with System/Data/Preboot/Recovery). A BaseSystem-only image can never boot
this way — no Preboot, no LocalPolicy, no `iBoot.img4`. It is attached **twice**:
as the root `pflash` (LLB reads boot objects through that) and as
`vmapple-virtio-blk-pci,variant=root`.

**The image's own NVRAM.** `aux.img` carries the boot variables *and* the
LocalPolicy iBoot checks. tart publishes it with a **16 KiB header**, so every
CHRP bank sits `0x4000` too high until it is stripped;
`fetch-tart-image.py` detects and removes it (and says so), because attached
unstripped the guest never finds its boot variables. A synthesised aux carrying only LLB makes LLB reload
itself forever (measured: 226 identical `type illb` iterations).

**The image's own ECID.** The on-disk LocalPolicy is personalised to it, so the
machine is started as `-M apple-vm,uuid=<ECID>`; `fetch-tart-image.py` prints
the ECID from the image's config layer.

### The one firmware patch

**The patch is not ours.** It comes from NyanSatan's
[Virtual-iBoot-Fun](https://github.com/NyanSatan/Virtual-iBoot-Fun), which
worked out what AVPBooter checks and which routine has to be neutered to run it
outside Apple's own hypervisor. Our contribution is only the script that applies
it safely to a firmware image you supply.

`scripts/patch-avpbooter.py` replaces the first two instructions of that routine
(`pacibsp; sub sp, sp, #0x10` at file offset `0x2314`) with `mov x0, #0; ret`.
That routine does not return success on this machine, and the boot stops before
LLB is loaded. Everything after it runs stock. The script checks the bytes at
that offset before writing, so a firmware build it does not recognise is refused
rather than corrupted.

### Boot arguments

There is no `-append` on this path: boot-args come from NVRAM. The aux holds two
CHRP banks (at `0xa00000` length `0x2000`, and `0xa80000` length `0x80000`),
each with an adler32 over `[base+0x14, base+banklen)` stored LE at `base+0x10`.
The live one is the bank with the higher generation.

**Append-only works; re-serialising the entry list does not.** A rewritten list
makes iBoot reject the NVRAM, set `iboot-failure-reason=0x65` and
`boot-command=recover`, and loop (measured: 523 iterations of stage 1, no stage
2). `scripts/prepare-aux.py` therefore appends one `KEY=VALUE\0` after the last
entry and fixes only that bank's checksum — 25 bytes differ for
`boot-args=-v serial=3`.

---

## 2. The CPU

The guest kernel is `RELEASE_ARM64_VMAPPLE` — built for Apple's *virtual*
machine, so it does not use the GXF/SPRR/APRR machinery a bare-metal Apple
kernel needs. What it does need is Apple's PAC model and a working PSCI.

### Pointer authentication is real, not a no-op

The stock kernel verifies its own signed pointers, so a no-op PAC cannot satisfy
it — it dies with *"JOP Hash Mismatch Detected (PC, CPSR, or LR corruption)"*.
`ORCHARD_REAL_PAUTH=1` selects QEMU's real implementation.

Apple's PAC differs from the architected one in three ways, all carried in
`target/arm`:

* a machine-wide **kernel key** mixed into every EL1 signature while
  `APCTL_KernKeyEn` is set;
* a **boot diversifier** (`keys.m`) XORed into key writes while
  `APCTL_AppleMode` is set, so the value the guest programs is not the value the
  hash sees;
* `APCTL_AppleMode` **enables the keys without the architected SCTLR bits**.

*Derived from [Inferno](https://github.com/ChefKissInc/Inferno) (ChefKissInc,
GPL-2.0-or-later): the `KERNELKEY*_EL1` / `APCTL_EL1` / `APCFG_EL1` registers,
the diversifier, and the two `pauth_helper.c` hunks. Everything else in this
section is ours.*

### Secondary cores come up keyless

Measured: the kernel programs the five key registers once, on the boot CPU, and
no secondary ever writes one. On real hardware a core leaves reset through
firmware that has already restored them; QEMU's PSCI `CPU_ON` jumps straight to
the kernel, so secondaries came up with **zero** keys — and every pointer the
boot CPU signed then failed to authenticate the moment another core touched it.
That is an FPAC fault whose own panic path re-faults, recursing until the kernel
stack is gone.

`ORCHARD_PAC_INHERIT=1` fixes it in two places: a snapshot of what each core
last programmed, reinstalled on reset, and the caller's keys handed to a core
brought up by `CPU_ON`. The CPU_ON half is restricted to a **caller at EL1 whose
pc is in the TTBR1 half** — that separates the kernel from a boot stage, and
handing a boot stage's secondary its caller's keys was measured to end in
*"Entering iBootStage1 recovery mode"* before the kernel ever loads.

### How many vCPUs

12, measured: desktop in ~75 s on the reference host.

There used to be a ceiling of 6. Above it the boot livelocked with zero disk
I/O, because XNU parks the cores it does not bring online in a `wfe; ldxr; cbnz`
pen and WFE is unmodelled under MTTCG, so each parked core burned a host CPU. A
WFE-parking patch was written for that and is **not in this tree**: the livelock
turned out to be fixed by `ORCHARD_PAC_INHERIT` — secondaries that come up with
valid keys leave the pen instead of faulting in it — and 12 vCPUs boot with no
WFE arm at all. One less divergence from upstream.

MTTCG costs still grow with vCPU count (every added core costs `CF_PARALLEL`
atomics on exclusive ops and turns each guest TLBI into an all-thread
rendezvous), so more is not automatically better; 12 is where this host
measured best.

---

## 3. The GPU

`reims-vgpu` ([steelbrain/reims-vgpu](https://github.com/steelbrain/reims-vgpu))
implements Apple's **ParavirtualizedGraphics** device: it decodes
the guest's command stream, translates Apple's AIR shaders to SPIR-V
(`metal2vulkan`), and renders on the host's Vulkan. The composited frame is
presented in its own window.

Two details shape everything else:

**Guest RAM is the GPU's memory.** The whole RAMBlock is imported once through
`VK_EXT_external_memory_host`, so a render target created over guest pages is
written by the host GPU directly — one copy of surface content, not two. This is
why `-m 8G` is a hard requirement rather than a preference: if the RAMBlock does
not fit the host's importable heap the import is refused
(`guest_ram_map_import_exceeds_heap`) and every bind falls back to a CPU byte
loader — slow, with black backgrounds.

**Plain anonymous guest RAM, not memfd.** A `memory-backend-memfd` RAMBlock
would let QEMU alias fragmented guest page runs into one contiguous host view
(`qemu_map_pages_alias_failed` disappears), but amdgpu's userptr rejects shared
file mappings, so the whole-RAM import then fails with
`ERROR_INVALID_EXTERNAL_HANDLE` — which costs far more than the aliases buy. The
alias failures in the log on an anonymous-RAM host are **expected**; those maps
take the copying rails.

---

### Where the firmware comes from, and why that matters

Upstream QEMU's vmapple documentation tells you to copy `AVPBooter.vmapple2.bin`
off a Mac, and that requirement is what made this route "needs Apple hardware to
set up" even though it runs on Linux. It does not: `Virtualization.framework`
lives in macOS, the macOS image is macOS, so the firmware is in the image.

Verified rather than assumed — the copy in the image and a copy from a Mac
differ by **six bytes at offset 657**, which is the version string
(`iBoot-8422.141.2.700.1` against a shorter one, zero-padded); the code is
byte-identical and the patch site at `0x2314` is the same. A boot from the
image-extracted firmware reaches the desktop in 105 s.

---

## What is ours

This project is not an assembly of other people's parts. Upstream QEMU can boot
a macOS 12 guest on a **macOS host with HVF**, where Apple's own hypervisor
supplies the CPU behaviour and Apple's ParavirtualizedGraphics framework
supplies the GPU. Neither exists on Linux. The work below is what stands
between that and a rendered desktop on a Linux/x86-64 box.

**Not the GPU.** `reims-vgpu` is
[steelbrain/reims-vgpu](https://github.com/steelbrain/reims-vgpu) — an
independent implementation of Apple's ParavirtualizedGraphics device, and the
single largest component here. It is included and wired into QEMU by this tree,
not written by it.

**Running Apple's firmware chain under emulation.** Getting `AVPBooter → LLB →
iBoot → XNU` to complete on TCG meant finding, one boot at a time, what the
chain requires: the disk attached both as pflash and as virtio so `fsboot` can
read boot objects; the image's *own* NVRAM, because the LocalPolicy is
personalised; the image's *own* ECID passed as the machine UUID; and NVRAM
edited append-only, because a re-serialised variable list makes iBoot set
`iboot-failure-reason=0x65` and loop forever. Each of those is a measured fact
with a failure mode attached, and none of them is documented anywhere.

**Apple's PAC under TCG.** The stock kernel verifies its own signed pointers, so
a no-op PAC is not an option. Beyond the register definitions (from Inferno),
ours is the part that makes it survive SMP: the per-core key snapshot restored
on reset, and PSCI `CPU_ON` key inheritance restricted to a kernel caller by
testing that its `pc` is in the TTBR1 half — because giving a *boot stage's*
secondary its caller's keys ends in iBootStage1 recovery before the kernel
loads. Without it, secondaries run keyless and the first cross-core pointer
authentication recurses through its own panic path until the stack is gone.

**The device model gaps.** Upstream's `bdif` silently drops NVRAM block writes,
so a guest can never persist a boot variable; ours implements the write path.
`cfg` needed a guest-visible random seed at a fixed offset. The GICv2m needed
bit 31 of `V2M_MSI_TYPER` before macOS would use the MSI frame at all. And the
machine needed Apple-style timer-on-FIQ wiring and a PAC master key installed
across CPUs.

**Portability work on upstream itself.** `CONFIG_VMAPPLE` was gated on HVF and
shipped off; `hw/vmapple/aes.c` contains a VLA that only ever compiled under
clang; the architectural `op1`-to-minimum-EL assertion rejects Apple's IMPDEF
registers. Those are upstream bugs or upstream assumptions that a Linux host
exposes for the first time.

**The operating knowledge.** Which of these matter is itself a result: 8 GB of
guest RAM because the RAMBlock must fit one Vulkan heap; plain anonymous RAM
rather than memfd because amdgpu's userptr rejects shared file mappings; 6
vCPUs because MTTCG's costs overtake the parallelism past that; and leaving
guest state alone across restarts because macOS finishes its install phases
across reboots.

---

## What we changed in QEMU

Against upstream QEMU master. 1 351 inserted lines across 15 files, plus 8 new
files.

| File | What and why |
|---|---|
| `hw/vmapple/vmapple.c` (+556) | The `apple-vm` machine: Apple-style timer-on-FIQ wiring, the reims-vgpu device, a PAC master key installed across CPUs, CHRP NVRAM building, and the GICv2m/MSI wiring newer guests need. |
| `hw/vmapple/bdif.c` (+456) | The backdoor interface: **NVRAM block writes** (upstream drops them, so the guest could never persist a boot variable), plus the virtual-USB queue lifecycle. |
| `hw/vmapple/cfg.c` (+18) | A guest-visible 64-byte random seed at the fixed offset `0x5c8`, and `production-mode` as a property. |
| `hw/vmapple/apple_vm_cpu.c` (new) | The `apple-vm-cpu` type: a thin subclass of `max` that selects real PAC, so VM quirks live here rather than in the shared ARM core. |
| `hw/display/reims-vgpu-*.c` (new) | The QEMU side of the GPU: MMIO and PCI transports, the dirty-page bridge, and the C↔Rust shim. |
| `target/arm/helper.c` (+148) | Apple PAC registers *(from Inferno)*; the per-core key snapshot and `arm_apple_pac_reset` *(ours)*. |
| `target/arm/arm-powerctl.c` (+43) | PSCI `CPU_ON` key inheritance, gated on an EL1 caller in the TTBR1 half. |
| `hw/intc/arm_gicv2m.c` | A `macos-compat` property that sets bit 31 of `V2M_MSI_TYPER`; macOS will not use the MSI frame without it. Off by default. |
| `target/arm/tcg/pauth_helper.c` (+15) | Kernel-key mixing and Apple-mode key enable *(from Inferno)*. |
| `target/arm/cpu.{c,h}`, `internals.h` | The state the above needs: kernel key, diversifier, `APCTL`/`APCFG`, and the reset hook. |

What we deliberately left out: the direct-kernelcache (`-kernel`) boot path from
the Inferno fork, which needs its Apple-silicon device-tree and Mach-O loader.
This tree boots the real firmware chain instead, so that dependency is gone.

---

## Porting notes

Upstream gates `CONFIG_VMAPPLE` on `depends on HVF` and ships
`CONFIG_VMAPPLE=n` for aarch64, because its own vmapple support assumes a macOS
host virtualising Apple silicon natively. This tree drops that dependency and
turns the machine on — running it under TCG on Linux is the point of the
project — and selects `REIMS_VGPU` where upstream selects `MAC_PVG_MMIO`.

Details the ported code had to be adapted to, since upstream has moved on since
the fork:

* headers moved: `hw/sysbus.h` → `hw/core/sysbus.h`, `block/aio.h` →
  `qemu/aio.h`, and similar for `boards`/`irq`/`loader`/`qdev-properties`/`usb`;
* the console and input APIs were renamed to what our compatibility macros
  already called them (`graphic_console_init` → `qemu_graphic_console_create`,
  `dpy_gfx_replace_surface` → `qemu_console_set_surface`, and so on), so the
  shims were deleted rather than extended; `qemu_input_event_send_key_qcode` is
  gone and `qemu_input_event_send_key_linux` does the evdev translation itself;
* `cpu_physical_memory_get_dirty_flag` is now `physical_memory_get_dirty_flag`
  in `system/physmem.h`;
* `CharBackend` is now `CharFrontend` (the `qemu_chr_fe_*` entry points keep
  their names);
* `TYPE_VMAPPLE_AES` is now `TYPE_APPLE_AES`, and `hw/arm/primecell.h` is gone;
* upstream's `define_one_arm_cp_reg` asserts that a register's declared access
  is no laxer than the architectural minimum EL implied by its `op1`. Apple does
  not follow that rule in its IMPDEF space — the kernel programs
  `KERNELKEY*_EL1` and `APCTL_EL1` (`s3_4_c15_*`, i.e. `op1=4`) from EL1 — so
  the mask for `op1` 4/5 is relaxed from `PL2_RW` to `PL1_RW`. Inferno relaxes
  it all the way to `PL0_RW`; EL1 is enough and keeps userspace out;
* the RAMBlock queries moved to `system/ramblock.h`. This one is worth stating
  plainly because it does not fail the build: with the declaration missing, C
  defaults `qemu_ram_block_from_host` to returning `int`, the `RAMBlock *` is
  truncated to 32 bits, and QEMU segfaults ~80 s into the guest boot the first
  time the device maps a fragmented page run. Configuring with
  `--disable-werror` turns that into a warning you will scroll past;
* `hw/vmapple/aes.c` declares `static const size_t MAX_LEN` and then sizes an
  initialised array with it, which is a VLA in C and which GCC refuses. It only
  ever compiled because upstream builds vmapple with clang on macOS; this tree
  makes it an `enum`. That one is worth sending upstream.

* `CPREG_FIELD32/64` are `#undef`ed immediately after `raw_read`/`raw_write` in
  `target/arm/helper.c`, so register accessors outside that region use
  `raw_read(env, ri)` / `raw_write(env, ri, value)`.

---

## Environment switches

Every switch the QEMU side reads, and nothing else — the diagnostic arms from
the bring-up (forced GFX stub, read-only NVRAM, forced dirty answers, QARMA
hashing, per-command tracers) were removed once they had answered their
questions.

| Variable | Effect |
|---|---|
| `ORCHARD_REAL_PAUTH=1` | Real pointer authentication instead of the no-op. Required: the stock kernel verifies its own signed pointers. Set by `run-vm.sh`. |
| `ORCHARD_PAC_INHERIT=1` | A core brought up by PSCI `CPU_ON` inherits the caller's PAC keys, and a reset restores what that core last programmed. Required for SMP. Set by `run-vm.sh`. |
| `ORCHARD_DIRTY_DEBUG=1` | Prints the dirty-tracking harvest split. Diagnostic; no effect on behaviour. |
| `REIMS_VGPU_WINDOW=1` | The GPU opens its own window. Set by `run-vm.sh`. |
| `REIMS_VGPU_FORCE_SCANOUT=1` | Also enqueue the scanout framebuffer, not only the composited rail. This is what shows the *early boot* picture — the Apple logo and progress bar are drawn through the framebuffer by iBoot and the kernel, before the guest's graphics stack produces a composited frame. Set by `run-vm.sh`; it is the one change this tree carries against `reims-vgpu`. |

`reims-vgpu` reads further `REIMS_VGPU_*` variables of its own; see that
project.

## Known problems

These are open, measured, and not worked around. A run that hits them is
behaving as expected for this tree.

### The guest restarts a few times before it settles

**Wait through 2–5 restarts on a fresh image.** macOS completes install and
update sequences *across* reboots (installer phases, MobileSoftwareUpdate,
firmware update) and records its progress in NVRAM and on disk. Until those
phases are satisfied the guest keeps restarting — and it converges only if the
state is left alone. Resetting the aux or the overlay between boots wipes the
progress and the guest retries phase 1 forever, which looks exactly like a
permanent reboot loop. Real hardware (and tart) persists both; so must you.

A restart takes 200–340 s of guest time and is kernel-initiated and orderly.

### A GPU page-table panic

Occasionally, with the desktop up, the guest panics:

```
panic(cpu N caller ...): "hitting assertion" @AppleParavirtPageTable.cpp:210
   apvAssert(vmNode->nodes[index])
```

Disassembled, line 210 is the release path refusing to clear a PTE word that is
already zero, in the Dock's GPU work. Either the guest released the same range
twice (its `release_pte` has no re-entry guard) or a host write reached a page
holding PTEs; the device's own `node_guard` alarm has never fired, but its
coverage is thin. Unresolved.

### Graphics defects

* **Anything with a corner radius fills black.** A partial draw whose colour
  attachment declares `DontCare` and for which no seed resolved keys
  `AttachmentLoadOp::DONT_CARE` against `initialLayout = UNDEFINED`, which
  licenses the driver to discard the contents — so texels outside the draw's
  scissor are lost. Measured on one boot: 718 such draws, ~5.0 M texels, every
  one of them `DontCare` into a guest-backed target. The repair is to preserve
  from the engine resident when it holds the target's contents; two attempts at
  it both hung the same conformance case and were withdrawn.
* **The wallpaper is striped** with 1px dark rows, periodic at ~8.5 rows. They
  are already present in the texture the guest hands the device, they are not
  in the artwork, and coverage, interpolation precision, compute, page
  boundaries and codec blocks have each been eliminated by measurement. Open.
* **Pointer input does not reach the guest.** Keyboard does.

### Host-side capture

There is no reliable way to screenshot the guest from the host on this setup:
the frame lives only in the device's Vulkan window, GNOME refuses screenshots to
non-portal clients, and PipeWire delivers no buffers for a static screen.
`REIMS_VGPU_PRESENT_DUMP=<path>` writes the exact frame the window presents and
is the instrument to use instead — a QMP `screendump` reads guest pages and
shows a *different* image.

### Performance, and the KVM question

On an x86-64 host this is TCG only — the guest is arm64, so there is nothing to
accelerate. Expect minutes to the desktop and a compositor that is usable but
not smooth.

**On an arm64 host, KVM is plausible but completely untested here.** What is
known rather than assumed:

* The guest kernel is `RELEASE_ARM64_VMAPPLE`, built for Apple's *virtual*
  machine. A survey of what it actually touches found **no unsupported system
  registers beyond four Apple `c15` ones** — the PAC registers this tree adds
  (`KERNELKEY*_EL1`, `APCTL_EL1`, `APCFG_EL1`). That is a short list, which is
  why the idea is worth stating at all.
* Those four are the problem. Under KVM the CPU executes guest instructions
  directly, so our `target/arm` emulation of them is inert: the *hardware* has
  to implement them. On an Apple-silicon host (Asahi Linux) it does, and the
  question becomes whether KVM traps and forwards them. On a generic arm64
  server — Graviton, Ampere, an ARM dev board — those registers do not exist
  and an access is an undefined instruction, so the guest would not survive its
  own PAC setup.
* Everything else in this tree is host-side and indifferent to the accelerator:
  the devices, the NVRAM handling, and the GPU (which renders through the
  host's Vulkan either way).
* The PSCI key inheritance is a TCG-path concern that KVM would make moot — real
  hardware brings cores up through firmware that has already restored them.

So: **untested**, with a plausible path on Apple-silicon-on-Asahi and a clear
reason to expect failure on a generic arm64 host. Nobody has run it; do not
read the above as a claim that it works.
