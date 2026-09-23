# modal-button-probe

Asks whether a guest modal's buttons actually reach the screen, and — when they
do not — which side dropped them.

## Why it exists

The bug it was written for is "the logout window sometimes shows no sleep,
shutdown and restart buttons, more likely when dark mode is on". That report is
a screenshot observation, and a screenshot on its own cannot say whether the
guest declined to draw the buttons or whether this device lost them on the way
to the screen. Those are different bugs with different owners, and no amount of
staring at the image separates them.

Two independent observations of the same frame do:

- **Intent** — the guest's accessibility API, asked directly what buttons
  `loginwindow`'s window has and at what rectangles. This reads the guest's own
  view hierarchy, upstream of anything this project touches.
- **Result** — the host-side capture of the QEMU window, measured at exactly
  those rectangles.

A button the guest declares and the frame does not show is a compositing loss
here. A button the guest never declares is a guest-side absence and not ours.
The probe prints which, per button, per trial.

## Usage

```sh
scripts/modal-button-probe/modal-button-probe.sh [-n TRIALS] [--appearance dark|light|alternate] [--keep DIR]
```

Needs a running guest reachable as `macos-vm` over SSH and a host QEMU window
the KDE screenshot script can capture. `--keep DIR` retains the PNG and the
accessibility dump of every trial, which is what you want the moment a trial
fails.

Exit codes: `0` every declared button was drawn, `1` at least one trial found a
declared-but-undrawn button, `2` setup failure (no guest, no window, the modal
never appeared).

## How the verdict is reached

Each button's rectangle is converted from guest points to capture pixels using
the scale read off the capture and the guest's reported display size — neither
is assumed, because the capture is downscaled to fit 720p and a guest at another
resolution would otherwise shift every rectangle silently.

The rectangle's greyscale standard deviation is then compared against a
background reference taken from the same frame: the modal is tiled with patches
and the *flattest* one is used. The dialog is mostly its own fill, so the
flattest patch is background — whatever the appearance and whatever the layout,
and without naming a corner that some dialog happens to put an icon in. A drawn
button has an edge and a label and varies more; a missing one leaves the fill.

There is no tuned threshold. Measured on the guest's logout modal, buttons read
sigma 0.055–0.104 against a background of 0.0008–0.0012 in both dark and light —
45x to 130x separation — so a bare `>` and any factor up to about 40 decide the
same cases.

## What it does not cover

It summons the **log-out** modal (`Log Out` / `Cancel`), because that one can be
raised from a script and dismissed again without acting on it. The dialog named
in the bug report is the Control-Power one (`Sleep` / `Restart` / `Shut Down` /
`Cancel`), which has no scriptable trigger that does not risk executing the
action. Both are windows of the same `loginwindow` process and go through the
same compositing path, so this is a proxy for that dialog and not the dialog
itself. Say "the logout modal" when reporting a result from it.

It also cannot see a button that is drawn in the *wrong place*: it only asks
whether something was drawn where the guest said the button is. A button
rendered 200 px away would read as MISSING at its declared rectangle, which is
the correct verdict for "the user cannot click it" but does not describe what
happened. Keep the frame (`--keep`) and look.
