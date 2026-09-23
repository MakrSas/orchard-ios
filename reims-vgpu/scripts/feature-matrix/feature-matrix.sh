#!/usr/bin/env bash
# Compile every supported reims-vgpu arm, tests included, plus the option ROM.
#
# The project supports three arms, one per host GPU API actually available:
# Metal on Apple, Vulkan through MoltenVK on Apple, and Vulkan on a native
# Linux ICD. QEMU's meson picks one per build and day-to-day work compiles one,
# so a rename or a cfg change could break another arm silently for days. This
# script is the gate.
#
# It checks `--all-targets`, not the bare default, so arm-specific test code
# compiles too. Compiling is not enough on its own: a test that compiles on an
# arm but is cfg'd out still tests nothing, so the script also reports how many
# tests each arm actually runs.
#
# It also gates formatting and rustdoc, both arm-independent and neither visible
# to anything else in the toolchain — rustc and clippy are silent on formatting,
# and rustdoc's broken-link lints are warnings that no ordinary build runs.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="${REPO}"
CROSS_TARGET="${CROSS_TARGET:-x86_64-unknown-linux-gnu}"
CARGO_CMD="check"
COUNT_TESTS=1

usage() {
  cat <<'EOF'
usage: scripts/feature-matrix/feature-matrix.sh [--build] [--no-counts]

Runs `cargo check --all-targets` (or `cargo build` with --build) over every
supported reims-vgpu arm and reports one PASS/FAIL line per arm. Exits non-zero
if any arm fails to compile.

It then reports, per natively-runnable arm, how many tests that arm actually
enumerates — a test that compiles but is cfg'd out still tests nothing, so a
cfg change that silently empties an arm shows up as a dropped count rather than
a green run. Counting links the test binaries, which is slower than checking;
pass --no-counts to skip it. The cross-compiled arm cannot be counted because
its binaries do not run on this host.

The supported arms:

  Metal              --features backend-metal,host-window         Apple only
  Vulkan / MoltenVK  --no-default-features
                       --features backend-vulkan,host-window      Apple
  Vulkan / native    same feature set                             Linux
  Metal + Vulkan     --features backend-metal,backend-vulkan,host-window
                                                                  Apple only

The last one is the reason this script has to keep growing cells. It carries
both rails in one binary and picks between them at run time (REIMS_VGPU_RAIL),
so it is the only cell where "which rail did this build compile" and "which rail
is running" are different questions. A `cfg` that conflates them compiles
cleanly on both single-rail cells and misroutes or silently drops work here.

A further cell checks crates/reims-vgpu-efi, the PCI option ROM every x86 boot
builds. It is a separate workspace targeting x86_64-unknown-uefi, so it is not
a backend arm — but it ships, and nothing else in the repository checks it.

Two cells before all of those run `cargo fmt --all -- --check`, once per
workspace. rustfmt.toml at the repo root is the format and both workspaces are
kept clean under it, so these are no-ops until a change leaves a diff.

The feature sets are exactly what vendor/qemu/hw/display/meson.build passes for
REIMS_VGPU_BACKEND=metal and REIMS_VGPU_BACKEND=vulkan. An Apple host builds
all three natively (the Linux one by cross check). A Linux host builds the
native Vulkan arm and cross-checks the other two.

The Metal arm cross-checks from any host. src/lib.rs rejects backend-metal on
`not(target_os = "macos")` — that is a condition on the *target*, not on the
host, so `--target *-apple-darwin` satisfies it and the real cfgs are
exercised. `cargo check` needs no Apple SDK. This script used to skip the arm
off Apple on the theory that Metal could not be cross-checked at all, and the
arm rotted to 11 errors unnoticed; every one of them was in first-party code
that this cell catches.

Checking is all that is claimed: it type-checks the arm, it does not link
against a real SDK and cannot run it.

Warnings do not fail an arm; the count is printed so drift stays visible.

env:
  CROSS_TARGET   Linux target to cross-check (default x86_64-unknown-linux-gnu)
  METAL_TARGET   Apple target to cross-check the Metal arm against off Apple
                 (default: aarch64-apple-darwin if installed, else
                 x86_64-apple-darwin)
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --build)
      CARGO_CMD="build"
      shift
      ;;
    --no-counts)
      COUNT_TESTS=0
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "feature-matrix: unknown argument '$1'" >&2
      usage >&2
      exit 64
      ;;
  esac
done

if ! command -v cargo >/dev/null 2>&1; then
  if [ -x "$HOME/.cargo/bin/cargo" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
  else
    echo "feature-matrix: ERROR: cargo not found" >&2
    exit 1
  fi
fi

HOST_TRIPLE="$(rustc -vV | awk '/^host: /{print $2}')"

if ! rustc --print target-list | grep -qx "$CROSS_TARGET"; then
  echo "feature-matrix: ERROR: unknown target '$CROSS_TARGET'" >&2
  exit 1
fi
if [ "$CROSS_TARGET" != "$HOST_TRIPLE" ] &&
  ! rustup target list --installed 2>/dev/null | grep -qx "$CROSS_TARGET"; then
  echo "feature-matrix: ERROR: target '$CROSS_TARGET' not installed" >&2
  echo "feature-matrix:        run: rustup target add $CROSS_TARGET" >&2
  exit 1
fi

# Feature sets, verbatim from vendor/qemu/hw/display/meson.build.
FEATURES_METAL="--features backend-metal,host-window"
FEATURES_VULKAN="--no-default-features --features backend-vulkan,host-window"
# Both rails in one binary, chosen at run time through REIMS_VGPU_RAIL. Apple
# only, because backend-metal needs target_os = "macos".
FEATURES_BOTH="--features backend-metal,backend-vulkan,host-window"

FAILED=0
RESULTS=()

# label, target triple (empty for host), feature args, then three optionals for
# a package that is not `reims-vgpu` in the workspace: its directory, its
# `-p`/`--manifest-path` selector, and its target scope.
run_cell() {
  local label="$1" target="$2" features="$3"
  local dir="${4:-$WORKSPACE_DIR}" pkg="${5--p reims-vgpu}" scope="${6---all-targets}"
  local target_args=()
  [ -n "$target" ] && target_args=(--target "$target")

  local log
  log="$(mktemp)"
  local status="PASS"
  # --all-targets is load-bearing: without it this compiles the product only,
  # and every arm's test code goes unchecked. The option ROM is the one cell
  # that cannot use it — its bin is `#![no_main]` with its own panic handler,
  # so libtest's harness collides with `std`'s on any target that has one.
  # shellcheck disable=SC2086  # $features, $pkg and $scope are argument lists.
  if ! (cd "$dir" && cargo "$CARGO_CMD" $scope $pkg \
    ${target_args[@]+"${target_args[@]}"} \
    $features --message-format short) >"$log" 2>&1; then
    status="FAIL"
    FAILED=1
  fi
  # cargo replays nothing for an up-to-date unit, so a cached cell reports no
  # warnings at all. Say "cached" rather than "0" — a silent 0 reads as clean.
  local warns
  if grep -q '^ *Checking reims-vgpu\|^ *Compiling reims-vgpu' "$log"; then
    warns="$(grep -c ': warning' "$log" || true)"
  else
    warns="cached"
  fi
  RESULTS+=("$(printf '%-4s %-46s warnings=%s' "$status" "$label" "$warns")")
  if [ "$status" = "FAIL" ]; then
    echo "--- $label ---" >&2
    grep -E ': error|^error' "$log" >&2 || cat "$log" >&2
  fi
  rm -f "$log"
}

# Formatting is one question for the whole tree, not one per arm, so it gets its
# own cell shape rather than a feature set. `cargo fmt --all -- --check` exits
# non-zero and prints the offending hunks; on a clean tree it is silent and
# costs a second. A missing rustfmt is a FAIL and not a SKIP: a gate that
# quietly stands down on the machine that lacks the tool is how the Metal arm
# rotted to 11 errors.
fmt_cell() {
  local label="$1" dir="$2"
  local log status
  log="$(mktemp)"
  if (cd "$dir" && cargo fmt --all -- --check) >"$log" 2>&1; then
    status="PASS"
    RESULTS+=("$(printf '%-4s %-46s %s' "$status" "$label" "clean")")
  else
    status="FAIL"
    FAILED=1
    RESULTS+=("$(printf '%-4s %-46s %s' "$status" "$label" "run: cargo fmt --all")")
    echo "--- $label ---" >&2
    head -40 "$log" >&2
  fi
  rm -f "$log"
}

# Every crate whose rustdoc must resolve. The replacement crates plus the rail:
# their documentation *is* where this project keeps its contracts, so a link
# that names nothing is a contract term pointing at nothing. `reims-vgpu` is
# deliberately absent — it carries the legacy device model and is not clean yet,
# and a cell that listed it would be a cell nobody could keep green.
DOC_CRATES=(
  reims-vgpu-wire
  reims-vgpu-protocol
  reims-vgpu-paging
  reims-vgpu-memory
  reims-vgpu-config
  reims-vgpu-observe
  reims-vgpu-core
  reims-vgpu-vulkan
  reims-vgpu-testkit
)

# Documentation, one question for the whole set rather than one per arm — none
# of these crates has a backend feature, so there is no arm for their docs to
# differ on. `-D warnings` is what makes it a gate: rustdoc's broken-link and
# private-link lints are warnings by default, so without it the check passes
# while the links stay dead. Shaped after `fmt_cell` for its reason too — a
# missing rustdoc is a FAIL, not a SKIP.
doc_cell() {
  local label="doc / replacement crates"
  local log pkgs=()
  local crate
  for crate in "${DOC_CRATES[@]}"; do pkgs+=(-p "$crate"); done
  log="$(mktemp)"
  if (cd "$WORKSPACE_DIR" && RUSTDOCFLAGS="-D warnings" \
    cargo doc --no-deps "${pkgs[@]}") >"$log" 2>&1; then
    RESULTS+=("$(printf '%-4s %-46s %s' "PASS" "$label" \
      "${#DOC_CRATES[@]} crates, links resolve")")
  else
    FAILED=1
    RESULTS+=("$(printf '%-4s %-46s %s' "FAIL" "$label" \
      "run: RUSTDOCFLAGS=-Dwarnings cargo doc --no-deps")")
    echo "--- $label ---" >&2
    head -40 "$log" >&2
  fi
  rm -f "$log"
}

# Enumerate an arm's tests without running them. `--list` makes the libtest
# harness print one `path::name: test` line per test and exit, so the count is
# what that arm would actually execute — cfg'd-out tests are simply absent.
# Only natively-runnable arms can be counted; a cross-compiled binary does not
# run on this host.
COUNTS=()
# label, feature args, then the same three optionals `run_cell` takes. A cell
# whose `all_targets` enumeration cannot link — the option ROM's — passes
# `--lib` as its scope and reports why rather than a misleading count.
count_cell() {
  local label="$1" features="$2"
  local dir="${3:-$WORKSPACE_DIR}" pkg="${4--p reims-vgpu}" scope="${5---all-targets}"
  [ "$COUNT_TESTS" -eq 1 ] || return 0
  local out lib total
  out="$(mktemp)"
  # shellcheck disable=SC2086  # $features and $pkg are argument lists.
  if ! (cd "$dir" && cargo test $pkg $features --lib -- --list) \
    >"$out" 2>/dev/null; then
    COUNTS+=("$(printf '%-46s %s' "$label" "(could not enumerate)")")
    rm -f "$out"
    return 0
  fi
  lib="$(grep -c ': test$' "$out" || true)"
  if [ "$scope" = "--lib" ]; then
    COUNTS+=("$(printf '%-46s lib=%-5s all_targets=%s' "$label" "$lib" \
      "(bin is UEFI-only)")")
    rm -f "$out"
    return 0
  fi
  # shellcheck disable=SC2086
  (cd "$dir" && cargo test $pkg $features -- --list) \
    >"$out" 2>/dev/null || true
  total="$(grep -c ': test$' "$out" || true)"
  COUNTS+=("$(printf '%-46s lib=%-5s all_targets=%s' "$label" "$lib" "$total")")
  rm -f "$out"
}

echo "[feature-matrix] host=$HOST_TRIPLE cross=$CROSS_TARGET cargo=$CARGO_CMD"

# Cells 0a and 0b — formatting, one per workspace. These run first because they
# are the cheapest and need no target installed, and because a formatting diff
# is the one failure a reviewer should never have to read a compile log to find.
fmt_cell "rustfmt / workspace" "$WORKSPACE_DIR"
fmt_cell "rustfmt / reims-vgpu-efi" "$REPO/crates/reims-vgpu-efi"

# Cell 0c — rustdoc over the replacement crates. Also arm-independent, and also
# a failure a reviewer should not have to read a compile log to find.
doc_cell

# Arm 1 — Metal. Native on Apple, cross-checked everywhere else: lib.rs gates
# backend-metal on target_os, so an Apple *target* is all the arm needs. Only
# the run half is Apple-only, which is why the off-Apple cell never counts
# tests.
case "$HOST_TRIPLE" in
  *-apple-*)
    run_cell "metal / $HOST_TRIPLE" "" "$FEATURES_METAL"
    count_cell "metal / $HOST_TRIPLE" "$FEATURES_METAL"
    ;;
  *)
    # arm64 macOS is the pathway this arm actually ships on, so prefer it and
    # fall back to the x86 Apple target; both carry target_os = "macos" and so
    # exercise the same cfgs.
    if [ -z "${METAL_TARGET:-}" ]; then
      for cand in aarch64-apple-darwin x86_64-apple-darwin; do
        if rustup target list --installed 2>/dev/null | grep -qx "$cand"; then
          METAL_TARGET="$cand"
          break
        fi
      done
    fi
    if [ -n "${METAL_TARGET:-}" ]; then
      run_cell "metal / $METAL_TARGET" "$METAL_TARGET" "$FEATURES_METAL"
      if [ "$COUNT_TESTS" -eq 1 ]; then
        COUNTS+=("$(printf '%-46s %s' "metal / $METAL_TARGET" \
          "(cross-compiled — cannot run here)")")
      fi
    else
      # Not a pass. Say which command restores the cell, because the last time
      # this arm went unchecked it accumulated 11 errors.
      RESULTS+=("$(printf '%-4s %-46s %s' "SKIP" "metal / $HOST_TRIPLE" \
        "(rustup target add aarch64-apple-darwin)")")
    fi
    ;;
esac

# Arms 2 and 3 — Vulkan through MoltenVK on Apple, native ICD on Linux. Same
# feature set; the host is what differs.
run_cell "vulkan,host-window / $HOST_TRIPLE" "" "$FEATURES_VULKAN"
count_cell "vulkan,host-window / $HOST_TRIPLE" "$FEATURES_VULKAN"

# Arm 4 — both rails in one binary, which is the configuration an Apple host
# uses to run one guest stream through Metal and through MoltenVK and tell a
# metal2vulkan defect from a defect in this device.
#
# It is not a fourth flavour of the other three; it is the cell that catches a
# `cfg` used to mean "which rail is running". Such a `cfg` compiles and passes on
# both single-rail cells and then silently drops or misroutes work here — which
# is exactly what `emit_object_cache_levels`, `mipmap` and `exec`'s preflight
# each did before the `Backend` trait took those decisions over. Cross-checked
# off Apple for the same reason arm 1 is: the target carries the cfgs.
case "$HOST_TRIPLE" in
  *-apple-*)
    run_cell "metal+vulkan / $HOST_TRIPLE" "" "$FEATURES_BOTH"
    count_cell "metal+vulkan / $HOST_TRIPLE" "$FEATURES_BOTH"
    ;;
  *)
    if [ -n "${METAL_TARGET:-}" ]; then
      run_cell "metal+vulkan / $METAL_TARGET" "$METAL_TARGET" "$FEATURES_BOTH"
      if [ "$COUNT_TESTS" -eq 1 ]; then
        COUNTS+=("$(printf '%-46s %s' "metal+vulkan / $METAL_TARGET" \
          "(cross-compiled — cannot run here)")")
      fi
    else
      RESULTS+=("$(printf '%-4s %-46s %s' "SKIP" "metal+vulkan / $HOST_TRIPLE" \
        "(rustup target add aarch64-apple-darwin)")")
    fi
    ;;
esac

# The supporting crates, counted separately because they are counted at all.
# `reims-vgpu`'s own count is the number this file has always printed, and a
# drop in it is supposed to mean a cfg change emptied an arm. When a module
# moves out into a crate its tests move with it, and a reader with only the
# first number would read that move as exactly the loss this cell exists to
# catch. Every arm links these, so the feature set does not change them.
count_cell "support crates / $HOST_TRIPLE" "" "$WORKSPACE_DIR" \
  "-p reims-vgpu-config -p reims-vgpu-memory -p reims-vgpu-observe \
   -p reims-vgpu-paging -p reims-vgpu-wire"

# The replacement crates. `reims-vgpu-core` is deliberately not a dependency of
# `reims-vgpu` yet — the replacement stays reachable only from model tests until
# production ingress switches — which means no arm above links it and nothing
# here would compile it. A gate that skips the code under construction is a gate
# that reports green on a tree that does not build, so it gets its own cell:
# checked with --all-targets so its tests compile, and counted so a cfg or a
# module move cannot empty it quietly.
run_cell "replacement crates / $HOST_TRIPLE" "" "" "$WORKSPACE_DIR" \
  "-p reims-vgpu-protocol -p reims-vgpu-core -p reims-vgpu-testkit -p reims-vgpu-vulkan"

# The decline vocabulary without the sink. `reims-vgpu-observe` is `no_std` plus
# `alloc` with its `std` feature off, so a layer below the device can name its
# own refusals in the same words the device logs them in. Nothing else in the
# matrix builds that configuration, and a `std::` that creeps into the
# vocabulary compiles fine everywhere else.
run_cell "observe / no_std vocabulary" "" "--no-default-features" \
  "$WORKSPACE_DIR" "-p reims-vgpu-observe" "--lib"

# And the library that depends on it, at `--lib` scope on purpose.
# `reims-vgpu-protocol`'s tests assert what a refusal renders as, which needs
# the sink, so they carry a dev-dependency on observe with `std` on. Under
# resolver 2 that feature is not unified into a build that does not compile
# tests — but `--all-targets` does compile them, so the cell above would let a
# `std::` into the library and never notice. This is the cell that notices.
run_cell "protocol / no_std library" "" "" "$WORKSPACE_DIR" \
  "-p reims-vgpu-protocol" "--lib"
count_cell "replacement crates / $HOST_TRIPLE" "" "$WORKSPACE_DIR" \
  "-p reims-vgpu-protocol -p reims-vgpu-core -p reims-vgpu-testkit -p reims-vgpu-vulkan"
if [ "$CROSS_TARGET" != "$HOST_TRIPLE" ]; then
  run_cell "vulkan,host-window / $CROSS_TARGET" "$CROSS_TARGET" "$FEATURES_VULKAN"
  if [ "$COUNT_TESTS" -eq 1 ]; then
    COUNTS+=("$(printf '%-46s %s' "vulkan,host-window / $CROSS_TARGET" \
      "(cross-compiled — cannot run here)")")
  fi
fi

# Arm 5 — the PCI option ROM. Not a backend arm and not a workspace member: it
# is its own workspace targeting x86_64-unknown-uefi, and `vm/boot-x86.sh`
# rebuilds it before every x86 boot. It was invisible to this script and to
# every command in AGENTS.md, which is the same gap that let the Metal arm rot
# to 11 errors: a live crate nothing checks.
#
# `--all-targets` cannot be used here. The bin is `#![no_main]` with the `uefi`
# crate's panic handler, so building its test harness collides with `std`'s —
# on the UEFI target and on the host alike. The lib is where the logic is
# (`paint`, the real Blt paths), and it does run on the host.
EFI_DIR="$REPO/crates/reims-vgpu-efi"
EFI_TARGET="x86_64-unknown-uefi"
if rustup target list --installed 2>/dev/null | grep -qx "$EFI_TARGET"; then
  run_cell "option-rom / $EFI_TARGET" "$EFI_TARGET" "" "$EFI_DIR" "" ""
else
  RESULTS+=("$(printf '%-4s %-46s %s' "SKIP" "option-rom / $EFI_TARGET" \
    "(rustup target add $EFI_TARGET)")")
fi
count_cell "option-rom / host lib" "" "$EFI_DIR" "" "--lib"

echo
for line in "${RESULTS[@]}"; do
  echo "[feature-matrix] $line"
done

if [ "${#COUNTS[@]}" -gt 0 ]; then
  echo
  echo "[feature-matrix] tests enumerated per arm:"
  for line in "${COUNTS[@]}"; do
    echo "[feature-matrix]   $line"
  done
  echo "[feature-matrix] A dropped count means a cfg change emptied an arm;"
  echo "[feature-matrix] compiling is not the same as testing."
fi

if [ "$FAILED" -ne 0 ]; then
  echo "[feature-matrix] FAILED: an arm does not compile, the tree is unformatted," >&2
  echo "[feature-matrix] or a documentation link resolves to nothing" >&2
  exit 1
fi
echo "[feature-matrix] all supported arms compile; both workspaces are rustfmt-clean;"
echo "[feature-matrix] the replacement crates' documentation links all resolve"
