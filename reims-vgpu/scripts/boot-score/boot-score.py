#!/usr/bin/env python3
"""Score a driven boot's fail log on CPU *and* GPU microseconds per draw.

Reads `/tmp/reims-vgpu-fail.log` (or the copies a harness keeps) and prints one
line per boot. It reads a log, never this repository's source; see `AGENTS.md`'s
ban on source-scanning gates for why that distinction is the whole licence for a
script like this to exist.

Two things it does that the obvious one-liner does not, and both were mistakes
made on real boots before they were fixed here.

**It joins the censuses by their own `t`, not by line ordinal.** `drain_duty`,
`gpu_span`, `window_publish` and `store_routes` are emitted from different places
and each skips windows the others do not — an idle second costs `gpu_span`
nothing and `drain_duty` a line. Pairing them by position therefore drifts within
a boot and pulls idle-desktop publishes into a driven band. The harness that had
been scoring driven Maps boots did exactly that and reported ~31 fps where the
same logs, joined by `t`, read 47-52.

**It reports the GPU half.** On the x86/Vulkan iGPU pathway `gpu_span busy_us`
per draw is *larger* than `drain_duty draw_us` per draw, and `(cpu + gpu) x
draws/s` accounts for 95-100 % of every busy second — the device is saturated and
the two halves barely overlap, so frames track their sum. A change ranked on
`us/draw` alone is ranked on half of itself.

Columns:

    n         busy census windows scored (drain duty >= 0.4, draws > 10)
    cpu       drain_duty draw_us / draws
    gpu       gpu_span busy_us / draws
    sum       the two added, which is what frames track
    fps       window_publish fresh / win_ms, over the same banded windows
    duty      mean drain_duty duty
    draws/s   mean draws per banded window
    occ       (cpu + gpu) x draws/s, the share of a second the two account for
    d/frame   draws per frame, which is the workload and drifts between boots

`occ` near 1.0 says both halves are the bottleneck and any microsecond off either
converts. Well under 1.0 says something else paces the guest and neither does.

`d/frame` is printed because frames are **not** comparable across boots without
it: `fps = 1e6 / (sum x d/frame)` and the second term is what the guest chose to
draw, which moves between boots of one binary.

Usage:  scripts/boot-score/boot-score.py FAIL_LOG [FAIL_LOG ...]
"""

import re
import sys

FIELD = re.compile(r"(\w+)=(-?[\d.]+)")
# Census windows are ~1 s apart and the emitters stamp `t` within a couple of
# milliseconds of each other, so a small window is an exact match and cannot
# reach a neighbouring second.
SLOP = 3


def _fields(line):
    return dict(FIELD.findall(line))


def score(path):
    gpu, pub, duty = {}, {}, []
    with open(path, errors="ignore") as handle:
        for line in handle:
            if line.startswith("OFF gpu_span "):
                f = _fields(line)
                if "t" in f and "busy_us" in f:
                    gpu[int(float(f["t"]))] = float(f["busy_us"])
            elif line.startswith("OFF window_publish "):
                f = _fields(line)
                if "t" in f and "fresh" in f and "win_ms" in f:
                    pub[int(float(f["t"]))] = (float(f["fresh"]), float(f["win_ms"]))
            elif line.startswith("OFF drain_duty "):
                duty.append(_fields(line))

    def near(table, t):
        for k in range(-SLOP, SLOP + 1):
            if t + k in table:
                return table[t + k]
        return None

    n = draws = cpu_us = gpu_us = duty_sum = fresh = win_ms = 0
    for f in duty:
        if "t" not in f:
            continue
        d, count = float(f["duty"]), float(f["draws"])
        if d < 0.4 or count <= 10:
            continue
        busy = near(gpu, int(float(f["t"])))
        if busy is None:
            continue
        n += 1
        draws += count
        cpu_us += float(f["draw_us"])
        gpu_us += busy
        duty_sum += d
        published = near(pub, int(float(f["t"])))
        if published:
            fresh += published[0]
            win_ms += published[1]

    if not n:
        return f"{path}: no joined busy windows — undriven, or a log from the test suite"
    per_second = draws / n
    total = (cpu_us + gpu_us) / draws
    fps = fresh / (win_ms / 1000) if win_ms else 0.0
    return (
        f"{path:<40} n={n:<3} cpu={cpu_us / draws:5.2f} gpu={gpu_us / draws:5.2f} "
        f"sum={total:5.2f} fps={fps:5.1f} duty={duty_sum / n:.2f} "
        f"draws/s={per_second:6.0f} occ={total * per_second / 1e6:.2f} "
        f"d/frame={draws / n / fps if fps else 0:6.0f}"
    )


def main(argv):
    if not argv:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    for path in argv:
        print(score(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
