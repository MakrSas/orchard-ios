// Exercises gpu-probe.html's frame-pacing statistics against synthetic frame
// streams whose answer is known by construction.
//
//   node scripts/browser-probe/pacing-stats-test.mjs
//
// WHY THIS EXISTS
//
// The pacing summary is an instrument, and an instrument that under-reports is
// worse than no instrument: a summariser that miscounted hitches would be
// indistinguishable from a guest that had none, and would be read as evidence
// that a frame-pacing goal was met. The cases below are the ones that would
// make it lie:
//
//   - a steady 60 Hz display must not be scored as dropping every frame, which
//     is what any hardcoded 120 Hz budget does;
//   - one long freeze and the same number of scattered stalls must not summarise
//     identically, because only the freeze is visible as a jerk;
//   - jitter inside the budget must not inflate the drop count.
//
// It parses the functions out of the page rather than restating them, so it
// tests the shipped code and not a copy that can drift from it.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, 'gpu-probe.html'), 'utf8');
const start = html.indexOf('const LONG_FRAME_INTERVALS');
const end = html.indexOf('function measureFps');
if (start < 0 || end < 0 || end <= start) {
  console.error('could not locate the pacing statistics in gpu-probe.html — ' +
                'the markers this test slices on have moved');
  process.exit(2);
}
// ESM is strict, so a bare eval keeps its declarations to itself; hand the one
// entry point back out explicitly.
eval(html.slice(start, end) + ';globalThis.summarizeDeltas = summarizeDeltas;');

let failed = 0;
const check = (name, got, want) => {
  if (JSON.stringify(got) === JSON.stringify(want)) {
    console.log(`ok   ${name} = ${JSON.stringify(got)}`);
  } else {
    failed++;
    console.log(`FAIL ${name}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`);
  }
};

const HZ120 = 1000 / 120;
const HZ60 = 1000 / 60;

// A display running exactly at its refresh rate. Every derived figure must say
// "nothing wrong here".
{
  const r = summarizeDeltas(Array(600).fill(HZ120), 5, 603);
  check('perfect.refresh_hz', r.refresh_hz, 120);
  check('perfect.long_frames', r.long_frames, 0);
  check('perfect.worst_hitch_frames', r.worst_hitch_frames, 0);
  check('perfect.dropped_refreshes', r.dropped_refreshes, 0);
  check('perfect.hist_intervals', r.hist_intervals, [600, 0, 0, 0, 0, 0]);
}

// Three isolated frames that each took two refreshes: three drops, but no run
// longer than one, so this is stutter rather than a freeze.
{
  const d = Array(600).fill(HZ120);
  for (const i of [100, 200, 300]) d[i] = HZ120 * 2;
  const r = summarizeDeltas(d, 5, 603);
  check('isolated.long_frames', r.long_frames, 3);
  check('isolated.dropped_refreshes', r.dropped_refreshes, 3);
  check('isolated.worst_hitch_frames', r.worst_hitch_frames, 1);
  check('isolated.refresh_hz', r.refresh_hz, 120);
}

// The same number of long frames, consecutive. `long_frames` cannot tell this
// apart from the case above; `worst_hitch_frames` is the field that can.
{
  const d = Array(600).fill(HZ120);
  for (let i = 300; i < 305; i++) d[i] = HZ120 * 2;
  const r = summarizeDeltas(d, 5, 603);
  check('freeze.long_frames', r.long_frames, 5);
  check('freeze.worst_hitch_frames', r.worst_hitch_frames, 5);
}

// One stall of ten refresh intervals: nine refreshes went unshown, and it lands
// in the saturating tail bucket.
{
  const d = Array(600).fill(HZ120);
  d[400] = HZ120 * 10;
  const r = summarizeDeltas(d, 5, 603);
  check('stall.dropped_refreshes', r.dropped_refreshes, 9);
  check('stall.hist_tail', r.hist_intervals[5], 1);
  check('stall.worst_frame_ms', r.worst_frame_ms, +(HZ120 * 10).toFixed(1));
}

// The regression that matters most: this rig's guest display has run at 60 and
// at 120. A 60 Hz stream is healthy and must not be reported as dropping.
{
  const r = summarizeDeltas(Array(300).fill(HZ60), 5, 303);
  check('sixty.refresh_hz', r.refresh_hz, 60);
  check('sixty.long_frames', r.long_frames, 0);
  check('sixty.dropped_refreshes', r.dropped_refreshes, 0);
}

// +/-1 ms of jitter around an 8.3 ms budget is not a skipped vblank.
{
  const d = Array.from({ length: 600 }, (_, i) => HZ120 + (i % 2 ? 1 : -1));
  const r = summarizeDeltas(d, 5, 603);
  check('jitter.long_frames', r.long_frames, 0);
}

// A run that is entirely long frames has no healthy median to compare against;
// it must not divide by zero or report a negative drop count.
{
  const r = summarizeDeltas(Array(120).fill(HZ60), 2, 123);
  check('uniform_slow.long_frames', r.long_frames, 0);
  check('uniform_slow.dropped_refreshes', r.dropped_refreshes, 0);
}

console.log(failed ? `\n${failed} check(s) FAILED` : '\nall checks passed');
process.exit(failed ? 1 : 0);
