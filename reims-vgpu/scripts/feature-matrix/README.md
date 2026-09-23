# feature-matrix.sh

Compile every supported `reims-vgpu` arm and fail if any of them does not build.

## Why

The project supports **three** arms, one per host GPU API actually available:

| Arm | feature set | host |
|---|---|---|
| Metal | `--features backend-metal` | macOS only |
| Vulkan / MoltenVK | `--no-default-features --features backend-vulkan,host-window` | macOS |
| Vulkan / native | same | Linux |

`vendor/qemu/hw/display/meson.build` picks the feature set from `REIMS_VGPU_BACKEND`, and
day-to-day work compiles exactly one arm. The crate spells
`all(feature = "backend-metal", target_os = "macos")` well over a hundred
times, so a rename or a `cfg` change can break an arm nobody builds and stay
broken indefinitely — which is what happened to the default build. Running this
script is part of "done" for any change that touches `cfg`-gated code.

There is deliberately **no** fourth arm. `backend-metal` off macOS has no Metal
to call, so it is a `compile_error!` in `src/lib.rs` rather than a host stub that
links and cannot draw.

## Run

```sh
scripts/feature-matrix/feature-matrix.sh              # check + per-arm test counts
scripts/feature-matrix/feature-matrix.sh --no-counts  # compile gate only (faster)
scripts/feature-matrix/feature-matrix.sh --build      # cargo build (link-level)
```

Output is one line per arm, then the test census:

```
[feature-matrix] PASS metal / aarch64-apple-darwin                   warnings=108
[feature-matrix] PASS vulkan,host-window / aarch64-apple-darwin      warnings=0
[feature-matrix] PASS vulkan,host-window / x86_64-unknown-linux-gnu  warnings=0

[feature-matrix] tests enumerated per arm:
[feature-matrix]   metal / aarch64-apple-darwin       lib=672   all_targets=685
[feature-matrix]   vulkan,host-window / aarch64…      lib=920   all_targets=992
[feature-matrix]   vulkan,host-window / x86_64…       (cross-compiled — cannot run here)
```

**Every arm checks from every host.** An Apple host builds the two Apple arms
natively and cross-checks the Linux one; a Linux host builds the native Vulkan
arm and cross-checks the two Apple ones. `src/lib.rs` rejects `backend-metal` on
`not(target_os = "macos")` — a condition on the **target**, not on the host — so
`--target aarch64-apple-darwin` satisfies it and the real `cfg`s are exercised,
and `cargo check` needs no Apple SDK to do it.

This paragraph used to say the opposite ("Metal is Apple-only and cannot be
cross-checked to a host that has no Metal"), which is the theory the script
itself abandoned after the arm rotted to 11 unnoticed errors — see the block
under `--help`. The script has cross-checked Metal from any host since; only
this note lagged, and while it lagged a session read it and concluded the Metal
arm was unverifiable from Linux. A gate nobody believes in is a gate nobody
runs.

Only *running* is host-bound, which is why a cross-checked arm reports
`(cross-compiled — cannot run here)` instead of a test count.

## Notes

- **Warnings do not fail an arm.** The count is printed on every run so a jump
  stays visible. cargo replays nothing for an up-to-date unit, so an arm that was
  already built reports `warnings=cached` rather than a false `0`.
- **The cross target defaults to `x86_64-unknown-linux-gnu`.** Override with
  `CROSS_TARGET=...`. The script fails with the `rustup target add` command when
  the target is missing rather than silently skipping the arm.
- **This is a compile gate only.** It says nothing about runtime behavior on any
  arm; a Vulkan change still needs a live boot on the host that owns the pathway.
- **It checks `--all-targets`, so the test modules are covered too.** A bare
  `cargo check` can leave arm-specific test code uncompiled while product code
  still compiles green.
- **Compiling is not testing, which is why the counts are printed.** A test that
  compiles on an arm but is `cfg`'d out still tests nothing, so a `cfg` change
  can empty an arm while every arm stays green. Watch for a *dropped* count. The
  gap between the arms (248 lib tests) is the genuinely Vulkan-only surface, not
  drift.
- **Counting links the test binaries**, which is slower than checking; use
  `--no-counts` when you only need the compile gate. The cross-compiled arm
  cannot be counted because its binaries do not run on this host.
