#!/usr/bin/env bash
#
# vm/rail-sweep.sh — boot every provisioned x86 rail, drive each desktop, and
# print the same set of readings from each.
#
# WHY THIS EXISTS. A defect found on one guest OS line is not a defect found on
# the pathway: `macos-26` refused 914 exec packets on `pipeline_absent` where
# `macos-15` refused none, and the withdrawal that fixed the pipeline race was
# provoked five times harder on 26 than on 15. `AGENTS.md` says not to
# generalise observations between guest rails, and this is what that costs in
# practice — a regression sweep means all six provisioned lines, not the one
# the last defect happened to appear on.
#
# It exists as a committed script rather than as a paragraph because the
# readings are only comparable if the workload and the key list are. Both live
# here: `vm/drive-desktop.sh` is the workload, and KEYS/SUMS below are the
# readings. Add a key here when a new route becomes part of what a sweep has to
# answer, so that every later sweep answers it too.
#
# WHAT IT REFUSES TO DO. It will not start while another boot is live. QEMU's
# `hostfwd tcp::2222` is a single host port, so a second boot dies instantly
# with "Could not set up host forwarding rule" — and then this script waits its
# full first-frame timeout for a frame that cannot come, which reads as a rail
# that failed to boot. `boot-x86.sh` also outlives its own QEMU, so killing
# QEMU alone does not free the port. The preflight below checks both.
#
# It also does not rebuild between rails, because `boot-x86.sh` does: editing
# source during a sweep gives the rails different binaries and the comparison
# they exist for is gone.
set -u
cd "$(dirname "$0")/.."

RAILS=("$@")
if [ ${#RAILS[@]} -eq 0 ]; then
  RAILS=(macos-11 macos-12 macos-13 macos-14 macos-15 macos-26)
fi

# Presence, not sums: a sum cannot establish a zero, because a route that never
# fired and a route that summed to zero print the same. `grep -c` on the route
# name answers "did this ever happen" and that is the question every key here
# is asking.
KEYS='packet_class_unclassified packet_unadmitted
      stamp_wait_model_ahead stamp_wait_model_behind stamp_publish_behind
      device_info_reply_frame_in_a_mapping device_info_reply_frame_in_no_mapping
      query_reply_inside_an_allocation backing_id_heap_placed
      exec_compute_record_unopened exec_segment_unended
      stream_frame_fail stream_record_fail
      pipeline_refuse_unnamed pipeline_advance_unnamed
      delete_unnamed_but_list_still_has_it observe_slug_collision'

SUMS='packet_class_exec,packet_class_lifecycle,packet_class_present,packet_class_query,packet_class_control,packet_class_unclassified,walk_records_render,walk_records_compute,walk_records_blit,pipeline_declared,pipeline_ready,pipeline_refused,pipeline_retired,pipeline_lease_withdrawn,pipe_memo_hit,pipe_memo_miss,preflight_mtlb_unloadable,stamp_published_by_channel,stamp_released_without_a_word,stamp_visible_observed,stamp_visible_inline,stamp_publish_behind,query_reply_scanned,query_reply_outside_every_allocation,device_info_reply_scanned,device_info_reply_frame_in_no_mapping'

# What is in the way, or nothing. **Kills nothing** — this is what the preflight
# asks, and a preflight that cleared the way instead of reporting it would
# destroy exactly the boot it exists to protect. Somebody else's evidence run is
# the common case for this being non-empty.
in_the_way() {
  pgrep -af "vm/boot-x86.sh|vendor/qemu/build/qemu-system-x86" | grep -v pgrep || true
  ss -ltn 2>/dev/null | grep ':2222 ' || true
}

# Stop the boot *this script* started, and wait for the port it held. Only ever
# called with a PID from this script's own `nohup`, so a sweep can never take
# down a boot it did not launch.
stop_our_boot() {
  local boot_pid=${1:-}
  [ -n "$boot_pid" ] || return 0
  # The QEMU first: it is what holds the port, and `boot-x86.sh` reverts its
  # snapshot clone once its child is gone, which is the tidy-up we want to run.
  pgrep -P "$boot_pid" -f "qemu-system-x86" | xargs -r kill 2>/dev/null || true
  kill "$boot_pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -q ':2222 ' || return 0
    sleep 1
  done
  echo "warning: host port 2222 still bound after stopping our boot"
  return 1
}

if [ -n "$(in_the_way)" ]; then
  echo "REFUSING TO SWEEP: a boot or its QEMU is live, or host port 2222 is"
  echo "bound. QEMU's ssh hostfwd is a single host port, so this sweep's first"
  echo "rail would die on it and then be reported as a rail that never"
  echo "presented a frame. Nothing has been killed — stop it yourself if it is"
  echo "yours, because it may be somebody else's evidence run."
  in_the_way
  exit 1
fi

BOOT_PID=""
trap 'stop_our_boot "$BOOT_PID"' EXIT

for rail in "${RAILS[@]}"; do
  echo "===== RAIL $rail $(date +%H:%M:%S)"
  stop_our_boot "$BOOT_PID" || { echo "RAIL $rail SKIPPED: previous rail's port is still held"; continue; }
  BOOT_PID=""
  rm -f /tmp/reims-vgpu-fail.log

  nohup ./vm/boot-x86.sh --device reims-vgpu-pci --testing --rail "$rail" \
    > "/tmp/sweep-$rail-boot.log" 2>&1 &
  BOOT_PID=$!
  ok=0
  for _ in $(seq 1 180); do
    grep -q "first frame presented" "/tmp/sweep-$rail-boot.log" 2>/dev/null && { ok=1; break; }
    sleep 5
  done
  if [ "$ok" = 0 ]; then
    echo "RAIL $rail NO FIRST FRAME"
    tail -5 "/tmp/sweep-$rail-boot.log"
    continue
  fi
  echo "first frame $(date +%H:%M:%S)"

  nohup ./vm/drive-desktop.sh > "/tmp/sweep-$rail-drive.log" 2>&1 &
  for _ in $(seq 1 120); do
    grep -qE "workload done|key auth failed" "/tmp/sweep-$rail-drive.log" 2>/dev/null && break
    sleep 5
  done
  echo "drive: $(tail -1 "/tmp/sweep-$rail-drive.log")"

  # Copied before anything else touches the guest: the next rail's boot clears
  # this file, and a reading taken after that is the next rail's.
  cp /tmp/reims-vgpu-fail.log "/tmp/sweep-$rail-fail.log"
  fail="/tmp/sweep-$rail-fail.log"

  echo "--- presence (0 = never fired) ---"
  for k in $KEYS; do printf "%-44s %s\n" "$k" "$(grep -c "$k" "$fail")"; done
  echo "rail_selected=$(grep -c rail_selected "$fail")  log_lines=$(grep -c . "$fail")"

  echo "--- sums ---"
  python3 - "$fail" "$SUMS" <<'PY'
import sys, re
path, keys = sys.argv[1], sys.argv[2].split(',')
# `store_routes` dumps are per-interval deltas carrying only the routes that
# fired in that interval, so a boot total is their sum and a single line is not.
total = {k: 0 for k in keys}
lines = 0
for line in open(path, errors='replace'):
    if 'store_routes' not in line:
        continue
    lines += 1
    for k in keys:
        for m in re.finditer(r'(?:^|[\s,])' + re.escape(k) + r'=(\d+)', line):
            total[k] += int(m.group(1))
print(f"  (dump lines: {lines})")
for k in keys:
    print(f"  {k}={total[k]}")
PY
done

stop_our_boot "$BOOT_PID" || true
BOOT_PID=""
echo "===== SWEEP DONE $(date +%H:%M:%S)"
