# Orchard iOS dependencies

Static libraries for arm64 iOS 16+, built from unmodified upstream sources
with the iPhoneOS SDK, for linking into Orchard's QEMU library. The same set
the Inferno-iOS toolchain builds.

| Library | Version | Licence | Source |
|---|---|---|---|
| GLib (with GIO, GObject, GModule) | 2.84.3 | LGPL-2.1-or-later | https://download.gnome.org/sources/glib/ |
| libffi (GLib subproject) | 3.2.9999 | MIT | https://github.com/libffi/libffi |
| PCRE2 (GLib subproject) | 10.44 | BSD-3-Clause | https://github.com/PCRE2Project/pcre2 |
| proxy-libintl (GLib subproject) | — | LGPL-2.1-or-later | https://github.com/frida/proxy-libintl |
| GMP | 6.3.0 | LGPL-3.0-or-later / GPL-2.0-or-later | https://gmplib.org/ |
| Nettle (with hogweed) | 3.10.2 | LGPL-3.0-or-later / GPL-2.0-or-later | https://www.lysator.liu.se/~nisse/nettle/ |
| pixman | 0.44.2 | MIT | https://cairographics.org/releases/ |
| libpng | 1.6.44 | libpng-2.0 | http://www.libpng.org/pub/png/libpng.html |
| libslirp | 4.9.1 | BSD-3-Clause | https://gitlab.freedesktop.org/slirp/libslirp |
| libtasn1 | 4.20.0 | LGPL-2.1-or-later | https://www.gnu.org/software/libtasn1/ |
| LZ4 | 1.10.0 | BSD-2-Clause | https://github.com/lz4/lz4 |
| LZFSE | git e634ca5 | BSD-3-Clause | https://github.com/lzfse/lzfse |
| libucontext | 1.5.2 | ISC | https://github.com/kaniini/libucontext |

zlib is the SDK's own; `zlib.pc` only points at it.
