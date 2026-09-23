# `runtime/census/` — declines that cannot be written where they happen

Three modules, each naming a class of loss that is otherwise invisible. They are not a measurement
tier: every one of them ends in an `observe::fail` or an `observe::Emit::decline(...).fail()`. What
puts them in their own directory is that the *reason* needs state the raising site does not hold —
a dedup set that spans draws, or a slug vocabulary shared by several call sites — so the line is
written here rather than inline.

## What is deliberately *not* here

`crate::observe` — the **sink**. `observe::fail` and `observe::off` are the always-on outputs that
everything here writes to, and the execution path calls them directly for its own declines. Filing
the sink under `census/` would suggest that a dropped guest command is a measurement. It is not: a
decline **must** be logged, a measurement **must not** change behaviour.

`observe::line` is the third tier — gated behind `REIMS_VGPU_DRAW_LOG=1`, so a failure logged only
through it is invisible on a normal boot and does not satisfy the fail-visible rule.

## The rule every module here obeys

**Measuring is allowed; branching on the measurement is not.** Nothing in the device or backend may
read one of these back to decide what to present, decode or execute. A proxy that changes behaviour
has stopped being a proxy and become a content heuristic, which the ground rules forbid outright.

## Reading them

They write to `/tmp/reims-vgpu-fail.log`. Every line here is deduplicated on the identity of the
thing that failed — `(reason, texture ref)`, `(site, format)`, `(reason, geometry)` — so the line
count measures *distinct* losses and never the frame rate. Zero lines is the healthy reading for all
three.

## Adding one

This directory has shrunk far more often than it has grown, and the test that shrank it is one
question: *if this were deleted, would a dropped guest command become invisible?*

- If the refusal already emits a typed decline at the point it refuses, what is left here only
  tracked its **rate**. Delete it; the decline is the report.
- A tally of **successful** work — binds that worked, frames that published, residents released,
  microseconds per sub-step — never had a claim under the rule at all.
- Cost hides behind "measure-only". Removed from this family: a GPU compute reduction per present, a
  registry scan under the engine lock per present, guest-descriptor reads per compute dispatch, and
  a second log file kept open for one event. Price a proxy at the rate it will actually run before
  writing `// Measure-only` on it.

And verify on a healthy boot that a new one fires **zero** times before calling the work done. A
proxy that floods is worse than none, because it trains the next reader to ignore the log — two
modules were deleted for emitting 15 310 lines across five boots, 99% of which said nothing had
happened.
