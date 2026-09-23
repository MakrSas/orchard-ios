#!/usr/bin/env bash
# collect-air.sh — keep every distinct AIR blob this boot translates, with the
# device timestamp it appeared at.
#
# `m2v_cache::translate_air` writes each blob it is about to translate to a fixed
# scratch name — `v.air`, `f.air`, `k.air` — in
# `/tmp/reims-vgpu-m2v-cache-<qemu pid>/`, holding only the **last** blob per
# stage. A boot translates dozens, so the file is a window and not a record. This
# polls it and keeps one copy per distinct content hash.
#
# Why the timestamp matters more than the blob. A translation failure can be
# found offline by sweeping the pool through the CLI, which is what
# `bugs/README.md` describes. A **runtime** failure cannot: the SPIR-V is valid,
# it compiles, and it then occupies the GPU past the host kernel's
# `preempt_timeout_ms` and gets the context reset. For that the useful question
# is *which shaders were translated just before the GPU stopped completing*, so
# each blob is filed with the wall clock at which it appeared and the device's
# own fail log supplies the other half of the correlation.
#
# The output directory holds Apple's AIR. It is under /tmp deliberately, and it
# must never be copied into this repository — see `bugs/README.md`.
#
# Usage:
#   scripts/app-sweep-probe/collect-air.sh --out DIR [--interval 0.1]
#
# Runs until killed. Start it before driving the guest.
set -uo pipefail
export LC_ALL=C

OUT=""
INTERVAL=0.1

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "collect-air: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] || { echo "collect-air: --out DIR is required" >&2; exit 2; }
# Every byte this writes is Apple's AIR, and the standing rule is that third-party
# bytes stay out of the index. `bugs/` is gitignored for the same reason, but a
# `--out` pointing anywhere else in the tree would not be, so the refusal is here
# rather than in a reviewer's memory.
mkdir -p "$OUT"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
case "$(cd "$OUT" && pwd -P)/" in
  "$REPO"/*)
    echo "collect-air: refusing --out inside the repository ($OUT) — this writes Apple's AIR, \
which must stay local and out of the index. Use a path under /tmp." >&2
    exit 2 ;;
esac
INDEX="$OUT/index.tsv"
: >"$INDEX"

# The device's own clock, so a blob can be placed against the fail log rather
# than against this script's start. `t=` in the log is milliseconds since device
# creation; the fail log's mtime moves with every line, so its first appearance
# is the closest available zero.
started=$(date +%s.%N)

seen=""
while :; do
  dir=$(ls -dt /tmp/reims-vgpu-m2v-cache-* 2>/dev/null | head -1)
  if [ -n "$dir" ]; then
    for stage in v f k; do
      src="$dir/$stage.air"
      [ -f "$src" ] || continue
      h=$(sha256sum "$src" 2>/dev/null | cut -c1-16)
      [ -n "$h" ] || continue
      case " $seen " in *" $stage:$h "*) continue ;; esac
      seen="$seen $stage:$h"
      now=$(date +%s.%N)
      rel=$(awk -v a="$now" -v b="$started" 'BEGIN{printf "%.3f", a-b}')
      cp -f "$src" "$OUT/$stage-$h.air" 2>/dev/null \
        && printf '%s\t%s\t%s\t%s\n' "$rel" "$stage" "$h" "$stage-$h.air" >>"$INDEX"
    done
  fi
  sleep "$INTERVAL"
done
