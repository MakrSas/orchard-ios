# Device config for the iOS cross build (--with-devices-aarch64=ios, or
# `aarch64-softmmu = 'ios'` in scripts/cross-ios-arm64.txt).
#
# Same as default.mak. REIMS_VGPU was off here while its Metal backend did not
# build for iOS; patches/reims-vgpu/0004 made it build for aarch64-apple-ios,
# and hw/display/meson.build selects backend-metal for an iOS host, so the
# guest's display is back on. The app reads the frames it puts in the console
# through ui/orchard-embed.c.

include ../arm-softmmu/default.mak

CONFIG_VMAPPLE=y
CONFIG_ARM_VIRT=y
CONFIG_REIMS_VGPU=y
