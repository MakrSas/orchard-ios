# Licence notice

* QEMU and our changes to it: **GPL-2.0-or-later** (`qemu/LICENSE`).
* `reims-vgpu/`: **LGPL-3.0** (`reims-vgpu/LICENSE`). Its author, Anees Iqbal
  (steelbrain), gave explicit permission on 2026-09-23 to use reims-vgpu in
  the Orchard app, superseding the licence in the repository — so that it can
  be linked with QEMU's GPL-2.0-only virtio code. Changes to reims-vgpu stay
  open here all the same, which is what its licence asks for.
* Apple PAC support in `target/arm` is derived from ChefKissInc/Inferno,
  GPL-2.0-or-later. See `ATTRIBUTION.md` for exactly which parts.
* The iOS app, `ios/`, and `qemu/ui/orchard-embed.c` with its header:
  **GPL-2.0-or-later**, like the rest of our QEMU changes. They began in
  Makr's Inferno-iOS (GPL-3.0 / AGPL-3.0 there) and are relicensed here by
  their author so that the app and QEMU can be one program.
* The iOS C dependencies (`scripts/fetch-ios-deps.sh`): each under its own
  licence, listed in `scripts/ios-deps-SOURCES.md`.

Neither Apple firmware nor macOS is distributed in this repository. See
`README.md` for what you must supply yourself and the licence terms that apply
to it.
