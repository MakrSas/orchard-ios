# web-content-probe

Checks whether the guest's web content reaches the screen intact, against a
declaration of what it was supposed to look like.

## Why it exists

Goal 8 is "web content is occasionally corrupted in Firefox and Safari —
background disappearing, subtle bugs, not all-white". A screenshot of a real
page cannot be checked against anything, because nothing says what it should
have looked like, and "occasional" means the interesting frame is one in
hundreds. Both problems are the same problem: there is no oracle.

So the page is built to be one. `content-probe.html` fills the viewport with a
fixed palette of widely separated, fully opaque colours in known regions, and
POSTs back each region's **screen-space** rectangle and the colour it is
supposed to be. The host then samples its own capture at exactly those
rectangles and classifies each measured mean to the nearest palette entry.

Same intent-versus-result shape as `scripts/modal-button-probe`, and for the
same reason: only the guest's own declaration can say whether a wrong pixel is
this device's fault.

## Usage

```sh
scripts/web-content-probe/web-content-probe.sh [-n CAPTURES] \
  [--browser safari|chrome|firefox] [--churn 0|1] [--keep DIR]
```

Needs a running guest reachable as `macos-vm` and a host QEMU window the KDE
screenshot script can capture. Exit `0` when every declared region measured its
declared colour in every capture, `1` on any mismatch, `2` on setup failure.
`--keep DIR` retains every frame and the layout records — do that whenever you
expect a failure, because the frame is the evidence.

## Churn, and why it witnesses itself

A static page is not the reported bug. "The background disappears" is reported
during real browsing — scrolling, layer promotion and demotion, elements
arriving and leaving — so `--churn 1` (the default) rebuilds a subtree of 24
rotated, half-promoted, scrolling children *behind* the declared patches every
beat. The patches are `position: fixed` and must be immune to all of it, so any
mismatch is a loss the churn caused rather than content that moved.

The churn itself is not checked region-by-region: it moves, and a moving region
measured at a fixed rectangle manufactures failures. But **one** churn child is:
`CHURN_WITNESS`, a `VIOLET` square placed at the crossing of the grid's centre
gaps, where no patch can occlude it, with its `top` compensating for the
container's scroll so its screen rectangle is the one the page published. It is
created and destroyed with the other 24 every beat, so it is a real churn child
rather than a bystander.

## How the verdict is reached

`WHITE` and `BLACK` are palette members on purpose: a region that loses its fill
classifies *by name* rather than as an unexplained threshold miss, so
"the background turned white" and "the background turned black" are different
verdicts. Nearest-entry classification also means there is no tolerance to tune
— the colours are far apart by construction, so "which of these is it" is
answerable without naming a distance that counts as close.

Each rectangle is inset by a sixth before measuring, so a one-pixel rounding
error in the 720p downscale cannot pull a neighbouring colour into the mean.

## The second mistake, which is the opposite of the first

The first churning run passed: six captures, zero mismatches. The retained
frames show **no churn children and no beat counter** — the whole output of the
repaint timer was missing, so it was six captures of a static page wearing a
churning run's label, and the verdict could not say so. The same page renders
churn correctly in a host browser, so this is about what the run could observe,
not about the page.

Where the first mistake was *the oracle disagreeing with a correct frame*, this
one is *the oracle agreeing with a frame that proves nothing*. A clean result
from a stressor that never ran is worth less than a failure, because it gets
recorded as evidence.

Two things now stop it, and they have to be in this order:

1. The page publishes its beat counter with every layout record, and the host
   **refuses to run** unless it advances (exit `2`, "the page's beat is
   stalled"). A capture whose beat did not move since the last one is not
   counted either way.
2. `CHURN_WITNESS` above, so a churn that is running but not reaching the screen
   fails the ordinary verdict.

The gate has to come first because the witness cannot tell those two apart on
its own: a wedged page and a device that lost a layer both leave `VIOLET`
missing, and only one of them is this device's fault.

## The third mistake, and the one the code now catches by itself

A Firefox run reported seven regions corrupted in nineteen consecutive captures.
The frame showed a **"make Firefox your default browser" sheet**, and macOS draws
a sheet by dimming the window behind it. Every measured colour was its declared
colour times 0.5 to the last bit — `RED` 224→111, `YELLOW` 240→120, the near-black
16→8 alike.

What makes this worth detecting mechanically rather than by eye is which regions
*passed*. Halved `BG` and halved `GREEN` are exactly equidistant from their own
palette entry and from `BLACK`, so the nearest-entry tie-break let them through.
A uniform dim therefore reports as a **partial**, entirely plausible-looking
corruption — seven of eleven regions, stable across captures, which is precisely
what a real intermittent bug would not look like but a reader might accept.

So the run now fits a single scale `k` across every region by least squares. A
tight fit at `k < 0.9` means the whole frame is attenuated, which is a state of
the guest's screen and not a loss in this device: the capture is discarded with
`ATTENUATED`, and a run that is mostly attenuated exits `2`. On the frame above
that fit gives `k = 0.4996` with a worst residual of `3.0/255`. Losing one region
to black instead gives a worst residual of `187`, so the guard cannot swallow a
real local loss — which is the whole reason it is a fit and not a tolerance.

The sheet itself is now prevented (`browser.shell.checkDefaultBrowser=false`) and
any other one is dismissed with two Escapes before the fullscreen chord — a sheet
swallows that chord too, which is why the run stayed windowed for all twenty
captures.

## The mistake this probe made first, kept because it will recur

The first run reported all six patches and two backgrounds wrong, identically,
in all five captures — and the retained frame showed the page rendering
*perfectly*. The declaration was wrong, not the pixels.

`window.innerWidth/innerHeight` and `window.outerWidth/outerHeight` do not update
atomically. Read during Safari's fullscreen transition, `innerWidth` was already
1920 while `outerWidth` was still 1854, so the computed viewport origin came out
at `(-33, -184)` and shifted every rectangle a whole grid row off the content.

Two things follow, and both are in the code now. The page **refuses** a
declaration whose origin is negative or whose viewport does not fit the screen,
rather than publishing it; and it republishes every second, with the host
re-reading the newest record before every capture. That also makes the probe
robust to a window being moved or resized mid-run, which would otherwise report
as ten simultaneous corruptions.

The general lesson is worth more than the fix: **when the oracle and the result
disagree everywhere at once, suspect the oracle.** A real compositing loss is
local and intermittent. A uniform, perfectly reproducible disagreement across
every region is a coordinate bug.

## What it does not cover

It detects a region whose *colour* is wrong at its declared rectangle. It does
not detect content shifted by a small amount (the inset absorbs a few pixels), a
wrong gradient, wrong text, or a transient wrong frame that falls between two
captures. It uses plain opaque divs deliberately — the aim is to catch a
compositing loss, not to exercise one particular path into one.
