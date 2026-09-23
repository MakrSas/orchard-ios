# ios-metallib-probe

One question, on real hardware: **does an iOS Metal driver accept a metallib
that Apple's macOS Metal compiler produced?**

That question decides whether the iOS port of `reims-vgpu`'s Metal backend
produces a desktop or a blank screen. The arm64 macOS guest compiles its
shaders with its own macOS Metal compiler and hands `reims-vgpu` the finished
MTLB blobs; `backend/metal/function.rs` passes them straight to
`newLibraryWithData:`. On a Mac that is a macOS library loaded by macOS. On an
iPhone it is a macOS library loaded by iOS, and nothing short of a device
settles whether the loader minds.

Nothing else about the port is blocked on this. The backend cross-compiles and
links for `aarch64-apple-ios` today (the `target_vendor = "apple"` gates in reims-vgpu); this is the
one runtime unknown big enough to be worth answering before anything else.

## Run it

```
./fetch-samples.sh                 # puts the blobs under test in Samples/
xcodegen generate                  # writes MetallibProbe.xcodeproj
open MetallibProbe.xcodeproj       # set your signing team, run on the device
```

The report appears on screen, goes to the console, and has a Copy button.

There is also a host-side build of the same probe, which needs no phone:

```
swiftc -O Sources/MetallibProbe.swift cli/main.swift -o /tmp/mtlbprobe && /tmp/mtlbprobe
```

`fetch-samples.sh` prefers compiling `probe.metal` twice, once per target,
which gives two blobs differing only in platform. That needs the Metal
toolchain component (`xcodebuild -downloadComponent MetalToolchain`). Without
it, the script falls back to copying one macOS and one iOS metallib off this
machine, which is enough for a yes/no on the loader.

`Samples/` is gitignored on purpose: those are Apple's compiled shaders, and
third-party binaries do not belong in this repository.

## What it reports

Three separable outcomes, because "it didn't work" has causes that imply very
different amounts of work:

1. **load** — did `makeLibrary(data:)` return a library at all?
2. **functions** — does it expose the function names it should?
3. **pipeline** — does a pipeline state build from one?

Only (3) means the shader would really run. Stage (3) is attempted **only** for
`probe_add`, the kernel in this directory's `probe.metal`, and never for a
harvested system library: Metal answers a question about a function it dislikes
with `abort()` rather than a thrown error — reading `functionType` on a
function out of Apple's own `default.metallib` kills the process — and the load
answer is the one worth protecting.

Every macOS-stamped sample is also retried with its platform bytes rewritten to
the iOS values. If the original is refused and the rewrite loads, the fallback
is a two-byte edit rather than a recompile.

The report also prints the GPU's family, BC-texture support, argument-buffer
tier and unified-memory flag, since those were the other open questions for the
port and the device is the only thing that can answer them either.

## The MTLB header bytes this relies on

Recovered by surveying roughly 900 metallibs present on a development machine —
macOS system frameworks, the iPhoneOS SDK, and the iOS simulator runtimes — and
reading off which bytes covary with the platform each was built for. Byte
`0x0B` separated the three platforms with no overlap, and byte `0x05`'s high
bit tracked the same split:

| platform       | `[0x0B]` | `[0x05] & 0x80` |
|----------------|----------|-----------------|
| macOS          | `0x81`   | `0x80`          |
| iOS (device)   | `0x82`   | `0x00`          |
| iOS Simulator  | `0x87`   | `0x00`          |

Also read, and useful in the report: `[0x08]` is the AIR language version,
`[0x0C..0x0E]` and `[0x0E..0x10]` the target OS major and minor (they matched
this machine's macOS version and the SDKs' iOS versions exactly, which is what
gives confidence the rest of the parse is aligned).

**This is an inference from samples, not a documented Apple layout.** A parse
that disagrees with a real file means the parse is wrong, not the file.

## The answer, measured

Run on an **Apple A16 GPU** (iPhone 14 Pro / 15 class):

```
device:  Apple A16 GPU
families: apple6 apple7 apple8
BC texture compression: false
argument buffers tier:  2
max buffer length:      3221225472
unified memory:         true

system-macos.metallib
  header:  macOS os=27.2 airLang=9
  verdict: LOADED (4 functions)

--- bottom line ---
A macOS-built metallib LOADS on this device as-is.
Rewriting the platform byte to iOS makes it load.
```

**A macOS-built metallib loads on an iOS device unmodified.** The loader does
not enforce the platform stamp in either direction. The restamp fallback this
probe was built to evaluate is not needed.

The same run settled a second question the other way, and corrected a wrong
guess in the process. BC texture compression reads **false** on an A16 that
advertises `apple6 apple7 apple8`, while an M1 Mac advertising only
`apple6 apple7` reads **true**. BC tracks *Mac*, not the family ordinal — no
iPhone has it, regardless of vintage. A macOS guest believes it is on a Mac and
will send BC textures; the Vulkan rail already degrades gracefully there and
the Metal rail has no equivalent gate yet.

## What the host run established first



Running the CLI on an M1 Mac:

- An **iOS**-stamped metallib loads on macOS with every function intact.
  macOS's loader does not enforce the platform byte.
- Restamping in **either** direction also loads, so the container carries no
  checksum over those bytes and the two-byte edit is mechanically viable.

That is evidence about the *macOS* loader, and the direction that matters is
the opposite one. It does mean that a refusal on iOS would be a deliberate
asymmetry in Apple's loader rather than a format incompatibility — which is a
better starting position than the reverse, but it is not the answer.

## Reading the result

- **Loads as-is** — the biggest unknown in the port is closed. Go straight to
  running the VM. *(This is what an A16 reported.)*
- **Refused as-is, loads restamped** — the fallback is real. `metal2vulkan` is
  already in the dependency graph and already parses MTLB for the Linux
  MTLB→AIR→SPIR-V path, so the restamp has a home and is not a blind edit.
- **Refused both ways** — read the error text. This is then a multi-day
  problem, not an overnight one, and worth re-scoping before spending a night
  on it.

## Packaging note: do not ship a fat metallib

A fat (multi-target) metallib archive starts with `0xCAFEBABE`, which is also
Mach-O's `FAT_MAGIC`. Sideloaders walk an app bundle looking for Mach-O
binaries to sign, hit that magic on a `.metallib`, and fail — LiveContainer
reports `Invalid mach-o file!` and refuses to install. The file is fine; it is
simply not Mach-O.

`fetch-samples.sh` therefore flattens every sample to a single plain `MTLB`
blob and stores them with a `.mtlbsample` extension, so nothing walking the
bundle mistakes them for code.

On the device itself, the same refusal shows up in `reims-vgpu`'s own log as
`metal_function_library_create_failed reason=… mtlb_len=N`. That slug is the
first thing to grep for in a run that boots but never draws.
