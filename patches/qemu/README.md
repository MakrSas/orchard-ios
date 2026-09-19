# patches/qemu

`qemu/` in this repository is a full QEMU source tree with our changes already
applied, so nothing here has to be applied to build it.

`0001-orchard-vmapple-on-linux.patch` is that same work as a diff against the
upstream commit the tree was taken from:

    base: f8aef8a9aed7438083c400da10acabdec485dc9b  (qemu/qemu master)

It exists so the delta stays legible — for review, for rebasing onto a newer
QEMU, and for sending the parts that belong upstream (the `hw/vmapple/aes.c`
VLA fix is one) to qemu-devel.

## One deliberate divergence from upstream

`qemu/tests/keys/` is not in this repository. It holds upstream's SSH test
fixtures (`id_rsa` and friends, used by the SSH block-driver tests), and GitHub's
secret scanning blocks a push that contains a private key — correctly, since it
cannot know the key is a public fixture. Nothing in this project builds or runs
those tests. Restore the directory from upstream if you need them.
