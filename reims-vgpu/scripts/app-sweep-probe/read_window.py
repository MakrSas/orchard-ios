#!/usr/bin/env python3
"""Reduce one slice of the fail log to the six numbers app-sweep-probe judges on.

Prints, space-separated on one line, in this order:

    gap_s  cadence_lines  present_hz_min  present_hz_med  drain_lines  alarms

`gap_s` is the widest interval between consecutive `drain_duty` censuses, in
seconds, taken from each line's own `t=` (device milliseconds since start) and
never from this host's clock — AGENTS.md is explicit that the two are different
clocks. It is the freeze signal: every census here is written at the *end* of a
drain tranche, so a drain thread that never returns emits nothing at all while
`display_vbl` and `host_window_loop` keep ticking from other threads and the
boot reads healthy.

A slice holding fewer than two censuses cannot have a gap measured, so it
reports the slice's whole span instead — which is the honest answer for a window
in which the drain produced one line or none, and is what makes `drain_lines=0`
read as a freeze rather than as a clean zero.

`alarms` counts the two lines that come from *outside* the drain and so survive
a wedged one: `driver_call_outstanding` (a host driver call past its deadline)
and `driver_quarantine` (a call a previous process died inside). Ordinary
fail-channel records are not counted — AGENTS.md warns that a named reason on
the fail channel is not automatically lost work, and several report a repair
that succeeded.
"""

import re
import statistics
import sys


def main(path: str) -> int:
    try:
        text = open(path, errors="replace").read()
    except OSError:
        print("0 0 0 0 0 0")
        return 0

    drain_t: list[int] = []
    span: list[int] = []
    hz: list[float] = []
    cadence = 0
    alarms = 0

    for line in text.splitlines():
        m = re.search(r"\bt=(\d+)\b", line)
        if m:
            span.append(int(m.group(1)))
        if " drain_duty " in f" {line} ":
            if m:
                drain_t.append(int(m.group(1)))
        if " host_window_cadence " in f" {line} ":
            cadence += 1
            h = re.search(r"\bpresent_hz=([0-9.]+)", line)
            if h:
                hz.append(float(h.group(1)))
        if "driver_call_outstanding" in line or "driver_quarantine" in line:
            alarms += 1

    if len(drain_t) >= 2:
        drain_t.sort()
        gap_ms = max(b - a for a, b in zip(drain_t, drain_t[1:]))
    elif span:
        # One census or none. The widest interval the slice can attest to is the
        # slice itself, and reporting 0 here would turn a wedged drain into a
        # clean reading.
        gap_ms = max(span) - min(span)
    else:
        gap_ms = 0

    print(
        f"{gap_ms / 1000:.1f} {cadence} "
        f"{min(hz) if hz else 0:.1f} {statistics.median(hz) if hz else 0:.1f} "
        f"{len(drain_t)} {alarms}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/dev/null"))
