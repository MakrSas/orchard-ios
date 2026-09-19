#!/usr/bin/env python3
"""Fetch a Cirrus Labs `tart` macOS image and lay it out for QEMU.

The image is an OCI artifact on ghcr.io. Three of its parts matter here:

  * `application/vnd.cirruslabs.tart.disk.v2` — the installed macOS disk, split
    into LZ4-compressed 512 MiB chunks. Written into one sparse raw file.
  * `application/vnd.cirruslabs.tart.nvram.v1` — the VM's NVRAM, which carries
    the boot variables *and* the LocalPolicy that iBoot checks. A synthetic
    NVRAM cannot replace it: the policy on the disk is personalised to this
    VM's ECID, so the aux has to be the one that shipped with the disk.
  * `application/vnd.cirruslabs.tart.config.v1` — the VM config, whose `ecid`
    is the identity the on-disk LocalPolicy was signed for. `run-vm.sh` passes
    it as `-M apple-vm,uuid=<ecid>`.

Everything is resumable: each chunk's download appends to a part file, and a
state file records the chunks already written, so an interrupted run continues
rather than restarting 30 GiB of transfer.

Copyright (c) 2026 Youssef Elliethy (yaelliethy)
SPDX-License-Identifier: GPL-2.0-or-later
"""

import argparse
import base64
import binascii
import hashlib
import json
import os
import plistlib
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    sys.exit("this script needs `requests` (pip install requests lz4)")

REGISTRY = "https://ghcr.io"
ZERO_64K = b"\x00" * 65536
CHUNK_SPAN = 536870912  # 512 MiB per disk layer, fixed by tart's own format


class TokenManager:
    """A ghcr pull token, refreshed before it expires."""

    def __init__(self, repo):
        self.repo = repo
        self.token = None
        self.expiry = 0
        self.lock = threading.Lock()

    def get(self, force=False):
        with self.lock:
            now = time.time()
            if not force and self.token and now < self.expiry - 60:
                return self.token
            url = f"{REGISTRY}/token?scope=repository:{self.repo}:pull"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            self.token = data["token"]
            self.expiry = now + data.get("expires_in", 300)
            return self.token


# Where the first CHRP NVRAM bank lives in the aux QEMU attaches, and the magic
# that identifies it.
NVRAM_BANK0 = 0xA00000
NVRAM_MAGIC = b"nvram"
TART_NVRAM_HEADER = 0x4000


def strip_nvram_header(raw):
    """Return the NVRAM as QEMU expects it, without tart's wrapper."""

    def bank_at(data, off):
        return data[off:off + 64].find(NVRAM_MAGIC) >= 0

    if bank_at(raw, NVRAM_BANK0):
        return raw
    if bank_at(raw, NVRAM_BANK0 + TART_NVRAM_HEADER):
        return raw[TART_NVRAM_HEADER:]
    # Neither layout: hand it over untouched rather than guess, and say so.
    print("WARNING: no CHRP bank at the expected offset; writing the layer "
          "as published")
    return raw


def manifest(tokens, repo, tag):
    headers = {
        "Authorization": f"Bearer {tokens.get()}",
        "Accept": "application/vnd.oci.image.manifest.v1+json",
    }
    url = f"{REGISTRY}/v2/{repo}/manifests/{tag}"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def blob(tokens, repo, digest):
    headers = {"Authorization": f"Bearer {tokens.get()}"}
    url = f"{REGISTRY}/v2/{repo}/blobs/{digest}"
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.content


def download_chunk(tokens, repo, digest, expected_size, part_path, idx):
    """Fetch one compressed layer, resuming a partial part file."""
    url = f"{REGISTRY}/v2/{repo}/blobs/{digest}"
    session = requests.Session()
    for _ in range(50):
        have = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        if have >= expected_size:
            return
        headers = {
            "Authorization": f"Bearer {tokens.get()}",
            "Range": f"bytes={have}-",
        }
        try:
            with session.get(url, headers=headers, stream=True,
                             timeout=(15, 60)) as resp:
                if resp.status_code in (401, 403):
                    tokens.get(force=True)
                    time.sleep(1)
                    continue
                if resp.status_code not in (200, 206):
                    time.sleep(2)
                    continue
                mode = "ab" if have else "wb"
                with open(part_path, mode) as out:
                    for piece in resp.iter_content(chunk_size=512 * 1024):
                        if piece:
                            out.write(piece)
        except (requests.RequestException, OSError) as exc:
            have = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            pct = (have / expected_size * 100.0) if expected_size else 0.0
            print(f"[chunk {idx:02d}] interrupted at {pct:.1f}%: {exc}; resuming",
                  flush=True)
            time.sleep(2)
    raise RuntimeError(f"chunk {idx} did not complete after 50 attempts")


def write_chunk(fd, idx, part_path, expected_size, expected_digest):
    """Decompress one LZ4 (Apple `bv4`) layer straight into the sparse image."""
    import lz4.block

    base = idx * CHUNK_SPAN
    hasher = hashlib.sha256()
    written = 0
    intra = 0
    data = open(part_path, "rb").read()
    off = 0
    dict_buf = b""
    while off < len(data):
        magic = data[off:off + 4]
        if magic == b"bv4$":
            break
        if magic == b"bv41":
            u_sz, c_sz = struct.unpack("<II", data[off + 4:off + 12])
            comp = data[off + 12:off + 12 + c_sz]
            plain = (lz4.block.decompress(comp, u_sz, dict=dict_buf)
                     if dict_buf else lz4.block.decompress(comp, u_sz))
            off += 12 + c_sz
        elif magic == b"bv4-":
            u_sz = struct.unpack("<I", data[off + 4:off + 8])[0]
            plain = data[off + 8:off + 8 + u_sz]
            off += 8 + u_sz
        else:
            raise ValueError(f"unknown magic {magic!r} in chunk {idx} at {off}")
        dict_buf = plain
        hasher.update(plain)
        # Holes stay holes: the image is 50 GB apparent and ~30 GB real.
        if plain != ZERO_64K and any(plain):
            os.pwrite(fd, plain, base + intra)
            written += len(plain)
        intra += len(plain)

    got = "sha256:" + hasher.hexdigest()
    if expected_digest and got != expected_digest:
        raise ValueError(f"chunk {idx} digest mismatch: {got} != {expected_digest}")
    if intra != expected_size:
        raise ValueError(f"chunk {idx} size mismatch: {intra} != {expected_size}")
    return intra, written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="cirruslabs/macos-ventura-base",
                    help="OCI repository (default: %(default)s)")
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--out-dir", default="images",
                    help="where disk.raw / aux.img / config.json land")
    ap.add_argument("--jobs", type=int, default=3,
                    help="parallel chunk downloads (default: %(default)s)")
    ap.add_argument("--metadata-only", action="store_true",
                    help="fetch the NVRAM and config layers and print the ECID, "
                         "but not the 30 GB disk")
    args = ap.parse_args()

    out = os.path.abspath(args.out_dir)
    parts = os.path.join(out, "parts")
    os.makedirs(parts, exist_ok=True)
    disk_path = os.path.join(out, "disk.raw")
    aux_path = os.path.join(out, "aux.img")
    cfg_path = os.path.join(out, "config.json")
    state_path = os.path.join(out, "fetch-state.json")

    tokens = TokenManager(args.repo)
    man = manifest(tokens, args.repo, args.tag)
    layers = man["layers"]

    disk_layers = [l for l in layers
                   if l.get("mediaType") == "application/vnd.cirruslabs.tart.disk.v2"]
    nvram_layers = [l for l in layers
                    if l.get("mediaType") == "application/vnd.cirruslabs.tart.nvram.v1"]
    cfg_layers = [l for l in layers
                  if l.get("mediaType") == "application/vnd.cirruslabs.tart.config.v1"]
    if not disk_layers:
        sys.exit("no tart disk layers in this manifest")

    # The config first: it is tiny and it carries the ECID the whole boot is
    # personalised to, so a run that stops here still tells you what to pass.
    if cfg_layers:
        cfg = blob(tokens, args.repo, cfg_layers[0]["digest"])
        open(cfg_path, "wb").write(cfg)
        try:
            ecid = json.loads(cfg).get("ecid")
            if ecid:
                pl = plistlib.loads(base64.b64decode(ecid))
                print(f"ECID: {pl.get('ECID')}  (pass as -M apple-vm,uuid=...)")
        except (ValueError, KeyError, plistlib.InvalidFileException,
                binascii.Error) as exc:
            print(f"WARNING: config saved but its ECID did not decode ({exc}); "
                  f"the boot needs one -- pass ECID=... to run-vm.sh")

    # The NVRAM: boot variables plus the LocalPolicy iBoot verifies.
    #
    # tart wraps it in a 16 KiB header, so the image as published is 0x4000
    # longer than the aux QEMU attaches and every CHRP bank sits 0x4000 too
    # high. Attaching it unstripped gets you a guest that never finds its boot
    # variables. The header is detected rather than assumed: the bank magic has
    # to appear at the expected offset once it is removed.
    if nvram_layers:
        raw = blob(tokens, args.repo, nvram_layers[0]["digest"])
        aux = strip_nvram_header(raw)
        open(aux_path, "wb").write(aux)
        print(f"aux (NVRAM): {aux_path} ({len(aux)} bytes"
              f"{', header stripped' if len(aux) != len(raw) else ''})")
    else:
        print("WARNING: no nvram layer; the boot will not get past iBoot")

    if args.metadata_only:
        sizes = [int(l.get("annotations", {})
                     .get("org.cirruslabs.tart.uncompressed-size", 0))
                 for l in disk_layers]
        print(f"disk: {len(disk_layers)} layers, "
              f"{sum(sizes) / (1 << 30):.1f} GiB uncompressed (not fetched)")
        return

    done = set()
    if os.path.exists(state_path):
        try:
            done = set(json.load(open(state_path)).get("completed", []))
        except (OSError, ValueError) as exc:
            print(f"WARNING: resume state unreadable ({exc}); "
                  f"re-fetching every chunk")
    lock = threading.Lock()

    def record(idx):
        with lock:
            done.add(idx)
            tmp = state_path + ".tmp"
            json.dump({"completed": sorted(done)}, open(tmp, "w"))
            os.replace(tmp, state_path)

    total_size = len(disk_layers) * CHUNK_SPAN
    fd = os.open(disk_path, os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size != total_size:
        os.ftruncate(fd, total_size)

    pending = [(i, l) for i, l in enumerate(disk_layers) if i not in done]
    print(f"disk: {len(disk_layers)} layers, {len(pending)} to fetch")

    def one(idx, layer):
        ann = layer.get("annotations", {})
        part = os.path.join(parts, f"chunk_{idx:02d}.part")
        download_chunk(tokens, args.repo, layer["digest"], layer["size"], part, idx)
        size, written = write_chunk(
            fd, idx, part,
            int(ann["org.cirruslabs.tart.uncompressed-size"]),
            ann.get("org.cirruslabs.tart.uncompressed-content-digest"))
        os.remove(part)
        record(idx)
        print(f"[{len(done):02d}/{len(disk_layers)}] chunk {idx:02d} "
              f"{size / (1 << 20):.0f} MiB ({written / (1 << 20):.0f} MiB non-zero)",
              flush=True)

    if pending:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(one, i, l) for i, l in pending]
            for fut in as_completed(futures):
                fut.result()
        os.fdatasync(fd)
    os.close(fd)

    st = os.stat(disk_path)
    print(f"disk ready: {disk_path}")
    print(f"  apparent {st.st_size / (1 << 30):.1f} GiB, "
          f"on disk {st.st_blocks * 512 / (1 << 30):.1f} GiB")


if __name__ == "__main__":
    main()
