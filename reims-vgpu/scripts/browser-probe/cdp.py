#!/usr/bin/env python3
"""Minimal Chrome DevTools Protocol client — no third-party packages.

The guest is an unmodified macOS install and the host is whatever the rig
happens to be; neither reliably has `websockets` or `websocket-client`, and
installing into the guest changes the snapshot we measure. RFC 6455 client
framing is ~80 lines, so this speaks it directly over `socket`.

Used to ask a browser questions that only the browser can answer — its own GPU
feature status, whether a WebGL/WebGPU context actually came up, and how many
frames it painted in a wall-clock second. Those are the measurements the
browser-facing goals are scored on; a screenshot cannot produce any of them.

Chrome and Firefox both expose CDP: Chrome natively on
`--remote-debugging-port`, Firefox from 129 onward behind
`--remote-debugging-port` + `remote.active-protocols=2`. Safari does not, and is
driven through `safaridriver` instead.

  ./cdp.py targets                     list debuggable targets
  ./cdp.py eval <url> <js>             navigate, evaluate, print the JSON result
  ./cdp.py text <url>                  navigate, print document.body.innerText
"""

import base64
import json
import os
import re
import socket
import struct
import sys
import time
import urllib.parse
import urllib.request

DEFAULT_ENDPOINT = os.environ.get("CDP_ENDPOINT", "127.0.0.1:9222")


class WebSocket:
    """Client half of RFC 6455, text frames only, no extensions negotiated.

    Fragmentation is handled on receive because CDP replies carrying a page's
    innerText routinely exceed one frame. Control frames (ping/close) are
    answered inline so a long `Runtime.evaluate` does not drop the connection.
    """

    def __init__(self, url, timeout=30.0):
        m = re.match(r"ws://([^:/]+):(\d+)(/.*)$", url)
        if not m:
            raise ValueError(f"not a wsurl: {url}")
        host, port, path = m.group(1), int(m.group(2)), m.group(3)
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode())
        self.buf = b""
        while b"\r\n\r\n" not in self.buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("handshake closed")
            self.buf += chunk
        head, self.buf = self.buf.split(b"\r\n\r\n", 1)
        if b"101" not in head.split(b"\r\n")[0]:
            raise ConnectionError(f"handshake refused: {head.splitlines()[0]!r}")

    def _recv_exact(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("socket closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def send(self, payload: str):
        data = payload.encode()
        header = bytearray([0x81])  # FIN + text
        n = len(data)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        mask = os.urandom(4)
        header += mask
        self.sock.sendall(bytes(header) + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def recv(self) -> str:
        parts = []
        while True:
            b0, b1 = self._recv_exact(2)
            fin, opcode = b0 & 0x80, b0 & 0x0F
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._recv_exact(8))[0]
            payload = self._recv_exact(length) if length else b""
            if opcode == 0x8:
                raise ConnectionError("peer closed")
            if opcode == 0x9:  # ping -> pong, then keep waiting for real data
                self.sock.sendall(b"\x8a\x80" + os.urandom(4))
                continue
            if opcode == 0xA:
                continue
            parts.append(payload)
            if fin:
                return b"".join(parts).decode("utf-8", "replace")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class Target:
    def __init__(self, ws_url, timeout=30.0):
        self.ws = WebSocket(ws_url, timeout=timeout)
        self.next_id = 1

    def call(self, method, params=None, timeout=30.0):
        mid = self.next_id
        self.next_id += 1
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
        raise TimeoutError(method)

    def evaluate(self, expr, await_promise=False, timeout=60.0):
        res = self.call(
            "Runtime.evaluate",
            {
                "expression": expr,
                "returnByValue": True,
                "awaitPromise": await_promise,
                "allowUnsafeEvalBlockedByCSP": True,
            },
            timeout=timeout,
        )
        if res.get("exceptionDetails"):
            raise RuntimeError(json.dumps(res["exceptionDetails"])[:800])
        return res.get("result", {}).get("value")

    def navigate(self, url, settle=3.0):
        self.call("Page.enable")
        self.call("Page.navigate", {"url": url})
        time.sleep(settle)

    def close(self):
        self.ws.close()


def http_json(endpoint, path):
    with urllib.request.urlopen(f"http://{endpoint}{path}", timeout=15) as r:
        return json.loads(r.read().decode())


def open_page(endpoint=DEFAULT_ENDPOINT, url=None):
    """Attach to a page target, opening a new tab when the browser allows it.

    Firefox's CDP shim has no `/json/new`, so fall back to whatever page target
    is already listed rather than failing the probe outright.
    """
    if url is not None:
        try:
            t = http_json(endpoint, "/json/new?" + urllib.parse.quote(url, safe=":/?&=#."))
            return Target(t["webSocketDebuggerUrl"])
        except Exception:
            pass
    pages = [t for t in http_json(endpoint, "/json/list") if t.get("type") == "page"]
    if not pages:
        raise RuntimeError("no page targets")
    tgt = Target(pages[0]["webSocketDebuggerUrl"])
    if url is not None:
        tgt.navigate(url)
    return tgt


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    endpoint = os.environ.get("CDP_ENDPOINT", DEFAULT_ENDPOINT)
    if cmd == "targets":
        for t in http_json(endpoint, "/json/list"):
            print(f"{t.get('type'):10s} {t.get('title','')[:60]:60s} {t.get('url','')[:80]}")
        return 0
    if cmd == "eval":
        tgt = open_page(endpoint, argv[2])
        try:
            print(json.dumps(tgt.evaluate(argv[3], await_promise=True), indent=2))
        finally:
            tgt.close()
        return 0
    if cmd == "text":
        tgt = open_page(endpoint, argv[2])
        try:
            print(tgt.evaluate("document.body.innerText"))
        finally:
            tgt.close()
        return 0
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
