#!/usr/bin/env python3
"""Serve the GPU probe page inside the guest and collect what it reports.

Runs inside the macOS guest, stdlib only.

The probe has to get its result out of a browser that may have no debugging
protocol (Safari) and no granted automation permission — AppleScript into
Safari or Finder times out on a fresh guest until someone clicks a consent
dialog, which an unattended run cannot do. Serving the page over HTTP and
letting it POST its own result back is the one channel that needs no consent
and works identically in Safari, Chrome and Firefox.

The page must be served over http rather than opened as a file:// URL, because
a file:// origin cannot POST to http://127.0.0.1.

  python3 probe_server.py <port> <html-path> <result-path>
"""

import http.server
import json
import os
import sys
import threading


def make_handler(html_path, result_path):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # the guest console is the serial log; keep it clean

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html", "/probe"):
                body = open(html_path, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                # A stale probe page is indistinguishable from a probe that did
                # not run, so never let the browser cache it.
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n)
            with open(result_path, "ab") as f:
                f.write(raw + b"\n")
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


def main():
    port, html_path, result_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    if os.path.exists(result_path):
        os.unlink(result_path)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), make_handler(html_path, result_path))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"serving http://127.0.0.1:{port}/ -> {result_path}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
