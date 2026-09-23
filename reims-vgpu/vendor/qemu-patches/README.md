# QEMU patches — vmapple bring-up for `vendor/qemu`

`vendor/qemu` is a **git submodule** (`git@github.com:steelbrain/qemu-reims-vgpu.git`, branch
`host-reims-vgpu-vmapple`). These patches are the **complete, re-applicable record** of
that branch relative to its branch-off base. They are not applied at build time
when the submodule already carries the commits; use them to rebuild the branch
from the base if needed.

## Base

```
b83371668192a705b878e909c5ae9c1233cbd5fb
```

Merge-base of the original `host-reims-vgpu-vmapple` cut from `origin/master` (not
latest master). All patches apply with `git am --3way` on a clean checkout of
that commit.

## Series

| Patch | What |
|-------|------|
| `0001` | RFC v7 patches 1–6 rebased: vmapple aarch64, apple-gfx, gicv2m, `CONFIG_VMAPPLE=y` |
| `0002` | HVF: skip `ARM_CP_NO_RAW` cpregs in `hvf_arch_init_vcpu` (macOS 26 host) |
| `0003` | HVF: emulate ISV=0 data aborts (LDP/STP, SIMD&FP, indexed LDR/STR) |
| `0004` | HVF: place in-kernel vGIC at the machine's GIC bases (`get_vgic_bases`) |
| `0005` | vmapple: `gfx-device` property (apple-gfx-mmio only) for Reims VGPU slots |

Rebuild:

```sh
git checkout b83371668192a705b878e909c5ae9c1233cbd5fb
git am vendor/qemu-patches/000*.patch
# tree must match host-reims-vgpu-vmapple tip
```

## Provenance

- `rfc-v7-vmapple-latest-macos.mbox` — upstream RFC v7 series. Patches 1–6 are carried here;
  patch 7 is excluded.

## Non-goals

- Rebase onto current `origin/master` (separate effort).
- Add product device logic to the patch series; that logic belongs in the submodule branch and the
  Rust crate.
