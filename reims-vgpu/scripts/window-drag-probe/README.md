# window-drag-probe

What this device does while a large window is being moved across the screen.

## Why it exists

Goal 6 is "window-dragging performance of Safari should be a stable 120 fps",
and nothing measured it. `scripts/browser-probe` measures rAF inside a page,
which is the wrong instrument twice: a window move is window-server compositing
rather than page script, and AGENTS.md records that Safari's rAF here is bimodal
at ~59 and ~118 with nothing between, so one rAF figure cannot support a claim
in either direction.

The device-side counters do not share that bimodality, so they are the result
and this script is the harness that makes them mean something. It marks the fail
log by byte offset, runs the motion, and reports `host_window_cadence`,
`drain_duty` and `readback_split` over exactly that window — counting **seconds
below 100 Hz** rather than a median, because a "stable 120" that spends a second
at 60 is not stable and a median hides precisely that.

## The two motions, and why the default is the weaker one

`--motion drag` posts a real Quartz `LeftMouseDown`/`Dragged`/`Up` stream
(`drag.c`, built on the guest, never shipped as a binary). That is what goal 6
names, and a pointer held across a title bar does not take the same path through
the window server as a window teleported by the accessibility API.

**It does not work on this guest, and the failure is silent.** `CGEventPost` to
the HID tap is discarded for a process that is not trusted for Accessibility.
Trust cannot be arranged here: there is no passwordless sudo and SIP's Filesystem
Protections are on, so `TCC.db` is unwritable. Measured: 1800 events posted at
exactly 120.0 Hz with zero late, and the window did not move one pixel.

So `--motion reposition` is the default. It moves the window through System
Events, which *is* trusted, at ~113 Hz. It is a weaker stressor — the window
server sees a sequence of window moves rather than a drag session — and it is
the default only because it is the one that runs. Say which motion produced any
number you quote.

## The guard that makes a result mean anything

The harness samples the window's real position mid-run and **refuses a verdict**
(exit 2) if it never left where it started. Without it the drag mode above would
have reported the idle device's counters as this device's ceiling, which is the
same failure `scripts/web-content-probe` made when it passed a static page off
as a churn test. The motion also reports what it actually did, so a run that
could not keep up says so instead of blaming the device.

## First measurement

x86/PCI guest, settled, Safari at 1000x640, `--motion reposition --seconds 15
--hz 120`. 1800 repositions in 15.9 s (113 Hz), window confirmed moved
(320,180) to (922,96).

```text
present_hz      n=15   min=0.40  med=10.90  max=12.60
duty            n=15   min=0.12  med=0.97   max=0.98
max_tranche_us  n=15   min=127054 med=128831 max=136075
draw_us/draw    n=15   min=106   med=109    max=237
flush_us/flush  n=15   min=741   med=946    max=2726
fence_us/fence  n=15   min=263   med=461    max=2135
seconds below 100 Hz: 15/15    worst second: 0.4 Hz
```

One representative second, verbatim:

```text
drain_duty win_ms=1095 tranches=47 busy_us=1016400 duty=0.928
  drain_us=1015528 max_tranche_us=129368 draw_us=320449 draws=2351
  flush_us=641149 flushes=523 max_flush_us=14971 slow_tranches=14/47
host_window_cadence window_ms=1005 presents=11 direct=11 offered=11
  present_hz=10.9 offered_hz=10.9 direct_frac=1.00
```

Read carefully, because the interesting parts are not the headline:

- The drain worker is busy for **essentially all** of each second (`duty` 0.93 to
  0.98) and produces **11 frames**. `offered` equals `presents`, so the device is
  not dropping frames it made — it made eleven.
- **`flush_us` is 641 ms of that second across 523 flushes.** The deferred
  writeback rail, which AGENTS.md already names the single largest cost in the
  device, is roughly two thirds of the worker's time under this workload.
- `max_tranche_us` is ~129 ms *every second*, with `slow_tranches=14/47`. A
  single tranche blocking the worker for 129 ms is fifteen frames at 120 Hz, and
  it is the shape of the hitching goals 5 and 6 describe.
- 2351 draws and 523 flushes for 11 presented frames is ~48 flushes per frame,
  which is the number worth explaining next.

## The control, and the defect it found

Reproduced on a second run of the same boot: `present_hz` med 10.65, `duty` 0.98,
`max_tranche_us` med 135112, 14/14 seconds below 100 Hz. So it is stable, not a
one-off.

The idle baseline — same guest, Safari open, no motion, 15 s — is the other half
of the claim:

```text
drain_duty win_ms=1002 tranches=249 busy_us=613 duty=0.001 max_tranche_us=8
  draw_us=0 draws=0 flush_us=0 flushes=0 slow_tranches=0/249
```

Zero draws, zero flushes, `duty` 0.001, worst tranche 8 µs. **The entire cost
above is caused by the motion; none of it is a standing cost of the device.**

Taking that control is also what exposed a defect in this harness. The first
attempt at it asked for "15 s at 12 Hz", and the reposition motion took `--hz`
as a step *count*: it ran 180 moves flat out and finished in 1.9 s, then
reported the counters as though they covered the intended fifteen seconds. The
loop is now duration-based, and `--hz` is documented as applying to `--motion
drag` only — System Events sets the rate here and the run reports what it
achieved. A knob that appears to be honoured and is not is worse than one that
is absent.

**What this does not establish.** The reposition stressor is synthetic and may
provoke more damage per move than a hand-driven drag; these numbers bound the
device under *this* workload, not under a user's drag. Nothing here separates
"the guest asked for this much work" from "the device is doing more than it was
asked". The `flush_us` share is consistent with the writeback ledger in
`flush_mapping_windows_before_fence` and is the first time that rail has been
seen dominating a *window-move* workload rather than a WebGL one.

Per AGENTS.md, record the host GPU's clock and power state beside any GPU-timing
number from this probe: on the measured host the governor accounted for six
sevenths of the fence wait.
