# Contributing

This tree is QEMU plus a narrow set of changes that make an arm64 macOS guest
boot on a Linux host. Before opening a patch, it helps to know which of three
places your change belongs in — most contributions that arrive here belong
somewhere else, and will get a better review there.

## Where does your change go?

| Change | Where |
|---|---|
| The GPU: rendering, shader translation, PVG protocol decode, anything under `reims-vgpu/` | Upstream: [steelbrain/reims-vgpu](https://github.com/steelbrain/reims-vgpu) |
| The AVPBooter patch itself — which routine to stub, a new firmware revision | Upstream: [NyanSatan/Virtual-iBoot-Fun](https://github.com/NyanSatan/Virtual-iBoot-Fun) |
| Generic QEMU: the ARM CPU model, block layer, TCG | Upstream: [qemu-devel](https://www.qemu.org/docs/master/devel/submitting-a-patch.html) — and please send it there rather than carrying it here |
| The `vmapple`/`apple-vm` machine, `hw/vmapple/*`, the QEMU side of the GPU device (`hw/display/reims-vgpu-*.c`), the scripts, the docs | **Here** |

Changes that belong upstream in QEMU are welcome here only as a temporary
carry, and each one should say in its commit message why it is not upstream yet.

## Building and running

```sh
scripts/build.sh                 # QEMU + the Rust device
scripts/fetch-tart-image.py --metadata-only --out-dir images   # no 30 GB, checks the registry path
scripts/boot-robust.sh
```

To build against a `reims-vgpu` checkout somewhere else, configure with
`--reims-vgpu-dir=/path`. It is resolved when the build is configured, not when
it runs, so exporting a variable before `ninja` does nothing.

`scripts/README.md` lists what each script does. You need your own
Apple's VM firmware comes out of the macOS image itself
(`scripts/extract-avpbooter.py`); no Mac is involved. See `LINUX.md`.

## The evidence rule

This is the one convention that matters here, because the whole project is
findings about a system nobody documents.

**A claim in a comment, a commit message or `TECHNICAL.md` carries the
measurement that produced it.** Not "this is faster", but "measured to splash:
1 vCPU 165 s, 4 vCPU 198 s, 8 vCPU 239–469 s". Not "the guest needs this", but
"without it: 523 iterations of iBootStage1, no stage 2". A reader has to be able
to contest your conclusion by re-running what you ran.

This applies to removals too: deleting a workaround means saying what you ran
that shows it is no longer needed. The WFE-parking patch was removed from this
tree exactly that way — measured at 12 vCPUs with the arm off, desktop in 75 s.

## Code

Follow QEMU's [coding style](https://www.qemu.org/docs/master/devel/style.html)
for anything under `qemu/`; the scripts follow PEP 8 and ordinary shell style.
Beyond that:

- **Comments explain why, never what.** If a comment restates the line below it,
  delete the comment.
- **No diagnostic switches in merged code.** Env-var arms for A/B experiments
  are how this code was built, but they do not ship: once the question is
  answered, the losing arm is deleted and the winner becomes the behaviour.
  `TECHNICAL.md` lists the four switches that survive; adding a fifth needs a
  reason in that table.
- **No speculative configuration.** No flag, option or parameter without a
  caller today.
- **Catch what you can handle.** Specific exception types in Python, checked
  returns in C; do not swallow an error to keep a path looking clean.

## Testing a change

There is no unit-test suite for the QEMU side — the test is a boot. For anything
touching the machine, the devices or `target/arm`:

1. `scripts/build.sh` must be warning-clean for the files you touched. Watch for
   `implicit declaration` in particular: with `--disable-werror` it is only a
   warning, and one of those truncated a pointer and segfaulted the guest 80
   seconds into a boot.
2. Boot to the desktop, and say in the PR how long it took and on what host.
3. If you changed the GPU device or the mapping paths, say whether
   `imported_mib` still matches the guest's RAM size — losing the zero-copy
   import is a large, quiet regression.

For a change to the boot chain or NVRAM handling, also boot **twice**: some
failures only appear on the second boot, because the guest persists state.

## Guest state, and why a fresh clone reboots a few times

macOS completes install and update sequences across reboots. A guest that
restarts two to five times on a fresh image is behaving normally; resetting the
aux or the overlay between boots wipes its progress and produces an endless
loop that looks like a bug in this tree. Do not "fix" a reboot loop by resetting
state — see *Known problems* in `TECHNICAL.md`.

## Legal

- Do not commit Apple binaries: no `AVPBooter`, no firmware, no macOS images,
  no extracted kernelcaches. `.gitignore` covers `images/`; keep it that way.
- QEMU changes are GPL-2.0-or-later. Sign off your commits
  (`git commit -s`) to certify the [DCO](https://developercertificate.org/).
- If you carry code from another project, say where it came from in the commit
  message and add it to `ATTRIBUTION.md`.

## Reporting a problem

Include the host (CPU, GPU, Mesa version), the guest build, how far the boot
got, and the tail of `serial.txt`. For a graphics problem, note that
`REIMS_VGPU_PRESENT_DUMP=<path>` writes the exact frame the window presented — a
QMP `screendump` reads guest pages and shows a *different* image, so the two
disagree and only the first one is evidence about what you saw.
