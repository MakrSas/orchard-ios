# Device config for the iOS cross build (--with-devices-aarch64=ios).
#
# Same as default.mak, minus REIMS_VGPU: its Metal backend isn't ready for
# iOS yet (in progress separately — see patches/reims-vgpu/). vmapple.c
# already falls back to a headless GFX/IOSurface MMIO stub when
# CONFIG_REIMS_VGPU is unset, so the machine still boots and is reachable
# over the serial console without it — good enough to validate the JIT/
# vmapple/PAC pipeline on-device before real graphics land.

include ../arm-softmmu/default.mak

CONFIG_VMAPPLE=y
CONFIG_ARM_VIRT=y
CONFIG_REIMS_VGPU=n
