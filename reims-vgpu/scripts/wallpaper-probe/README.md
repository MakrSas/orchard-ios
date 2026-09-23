# wallpaper-probe

Measures whether the desktop wallpaper reaches the screen where the guest put
it, and if not, by how much and in what shape.

## Why it exists

Goal 10 is "the wallpaper is sometimes shifted 10% to the left, so the right 10%
of the screen's background is all black — more likely when switching between
dark and light mode and between the different dynamic wallpapers".

A screenshot of a shifted wallpaper shows that something moved. It cannot say by
how much, whether the amount is the same at the top of the screen as at the
bottom, or whether the guest composited it that way. Those are the questions
that decide who owns the bug:

| what the bands say | what it means |
|---|---|
| same shift in all three | a uniform origin offset |
| shift growing down the screen | a row stride mismatch, and the per-band difference gives the stride error |
| no shift, bars lost at an edge | content clipped, not moved |
| every bar lost in every band | the desktop is covered — not a result |

So the probe supplies the wallpaper, which makes the declaration stronger than
the ones `scripts/modal-button-probe` and `scripts/web-content-probe` can get:
we do not have to ask the guest what it intended to draw, because we chose it.

## The image

64 vertical bars in two colours, in a fixed aperiodic pattern. Fixed rather than
generated per run so a decode can be reproduced from the log alone. It was
chosen by minimising agreement with every shifted copy of itself **over the
overlap** — which is how a shifted wallpaper actually presents, content sliding
with the vacated side lost, rather than wrapping. Worst non-zero-shift agreement
is `0.600` against `1.000` at zero shift, so an offset is never ambiguous.

Neither symbol is black or white. A bar that lost its fill therefore answers
"gone" instead of being rounded to whichever symbol it happens to be nearer,
which is the whole point of the reported black band.

## Usage

```sh
scripts/wallpaper-probe/wallpaper-probe.sh [-n TRIALS] [--keep DIR]
```

Needs a running guest reachable as `macos-vm` and a host QEMU window the KDE
screenshot script can capture. Exit `0` when every band of every trial decoded
at zero shift with no lost bars, `1` on any shift or loss, `2` on setup failure.

Each trial after the first perturbs first: it flips the appearance and passes
the desktop through one of the system's own dynamic pictures before coming back
to ours. That is the trigger the bug report names, and going away and returning
is also what defeats macOS's caching of the desktop picture by path — a plain
re-set of the same path does not necessarily redraw anything.

Before measuring, the probe asks the guest which picture it believes is on the
desktop. If the answer is not ours, the trial is skipped rather than judged: the
perturbation not having come back is a setup failure, not a corruption.

## Resolution, stated rather than implied

The decode is in whole bars, so on a 1920-wide desktop the shift it reports is
quantised to 30 px, and a sub-bar shift is rounded to the dominant symbol under
each bar. That is ample for a symptom reported at 10% of the screen (192 px, 6.4
bars) and it is not a sub-pixel instrument. A wrong *colour* with no
displacement, a vertical shift, and a wrong gradient are all outside what it
looks at.

## Verified against synthetic frames

Built the barcode, downscaled it to the capture size the KDE screenshot script
produces (1280x719 for a 1920x1080 guest), and ran the probe's own decoder:

| frame | top | mid | bot |
|---|---|---|---|
| unmodified | `shift=0 agree=1.00 lost=0` | same | same |
| slid left 192 px, black behind | `shift=-6 lost=6` | `shift=-6 lost=6` | `shift=-6 lost=6` |
| sheared 0 px at top to 240 px at bottom | `shift=-2 lost=2` | `shift=-4 lost=4` | `shift=-6 lost=6` |
| all black | `lost=64` | `lost=64` | `lost=64` |

The second and third rows are the discrimination the probe exists for: the
reported symptom and a stride bug produce the same screenshot to the eye and
different numbers here.
