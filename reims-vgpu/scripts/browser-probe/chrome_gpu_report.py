#!/usr/bin/env python3
"""Print Chrome's own GPU verdict, from `chrome://gpu`, as plain lines.

Runs inside the guest next to `cdp.py`. Kept as a file rather than a heredoc
piped through ssh: the report needs `\\n` inside a JS string literal, and a
double-quoted ssh command eats one level of backslash, which turns the
expression into a syntax error at a line number that points nowhere useful.

chrome://gpu renders its report inside a shadow root, so `document.body.innerText`
is empty and the tree has to be walked explicitly.
"""

import sys
import time

sys.path.insert(0, "/tmp")
import cdp  # noqa: E402  (path is set above)

DEEP_TEXT = """(() => {
  const out = [];
  const walk = (root) => {
    for (const el of root.querySelectorAll('*')) if (el.shadowRoot) walk(el.shadowRoot);
    out.push(root instanceof ShadowRoot ? root.textContent : (root.innerText || ''));
  };
  walk(document.body);
  return out.join(String.fromCharCode(10));
})()"""

FEATURES = (
    "Canvas:",
    "Compositing:",
    "OpenGL:",
    "Metal:",
    "Rasterization:",
    "WebGL:",
    "WebGL2:",
    "WebGPU:",
    "Video Decode:",
    "Video Encode:",
    "Skia Graphite:",
    "Direct Rendering Display Compositor:",
)

DRIVER = (
    "GL_VENDOR",
    "GL_RENDERER",
    "GL_VERSION",
    "Display type",
    "GL implementation parts",
    "GPU0",
    "Skia Backend",
    "GPU process crash count",
    "Initialization time",
    "Machine model name",
)


def main():
    tgt = cdp.open_page(url="chrome://gpu")
    time.sleep(4)
    txt = tgt.evaluate(DEEP_TEXT)
    tgt.close()
    if not txt:
        print("EMPTY chrome://gpu report", file=sys.stderr)
        return 1
    # The feature list is one text node with "*   " separating entries.
    for line in txt.replace("*   ", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(FEATURES):
            print("FEATURE  " + line)
        elif line.startswith(DRIVER):
            print("DRIVER   " + line[:220])
        elif "has been disabled" in line or "unable to boot" in line or "is unavailable" in line:
            print("PROBLEM  " + line.split("    Disabled")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
