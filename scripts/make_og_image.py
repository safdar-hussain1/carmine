"""Renders web/public/og-image.png, the picture shown when the site's link is shared.

LinkedIn, Slack, WhatsApp and X show this image in a link preview, so it is a
real screenshot of the page rather than a drawn card: the built site (docs/)
is served locally and opened in headless Chrome at 1600x840, the demo runs
the sample portrait by itself, and once its opening sequence has settled on
the before/after split the velvet look is picked and the viewport captured.

1600x840 has the 1200x630 shape the previews want but enough height for
the hero's whole composition; the capture is scaled down to 1200x630. The
full-bleed photograph does not fit the 500 KB limit as a lossless PNG, so
when it is over, it is reduced to a 256-colour palette with dithering,
which keeps the face and the lipstick smooth at preview size.

A screenshot goes stale when the page changes, which is why this is a script.
It captures docs/, and the image reaches docs/ through web/public/ on the
next build, so run it between two builds:

    (cd web && npm run build) && python scripts/make_og_image.py && (cd web && npm run build)

Needs Chrome or Chromium, and Pillow, which the dev extra installs (through
scikit-image).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from verify_site import (
    DOCS_DIR,
    REPO_ROOT,
    _DevToolsSocket,
    _find_chrome,
    _free_port,
    _page_tab,
    _serve_docs,
)

OUT_PATH = REPO_ROOT / "web" / "public" / "og-image.png"
WIDTH, HEIGHT = 1200, 630
CAPTURE_WIDTH, CAPTURE_HEIGHT = 1600, 840
MAX_BYTES = 500 * 1024

# The status line the mirror settles on once the sample is drawn and the
# opening sequence has finished.
SAMPLE_READY = "document.querySelector('.stage__status-text')?.textContent === 'Sample photo'"
CLICK_PRESET = (
    "[...document.querySelectorAll('.preset')]"
    ".find((b) => b.querySelector('.preset__name')?.textContent.trim() === {label!r}).click()"
)


class _Page:
    """One DevTools connection to the page, sending commands in order."""

    def __init__(self, ws_url: str) -> None:
        self._socket = _DevToolsSocket(ws_url, 120)
        self._next_id = 0

    def call(self, method: str, **params):
        self._next_id += 1
        message_id = self._next_id
        self._socket.send(json.dumps({"id": message_id, "method": method, "params": params}))
        while True:
            message = json.loads(self._socket.recv())
            if message.get("id") != message_id:
                continue  # an event, not our reply
            if "error" in message:
                raise RuntimeError(f"{method} failed: {message['error']}")
            return message.get("result", {})

    def evaluate(self, expression: str):
        result = self.call(
            "Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True
        )
        if "exceptionDetails" in result:
            raise RuntimeError(f"page evaluation failed: {result['exceptionDetails']}")
        return result.get("result", {}).get("value")

    def wait_for(self, expression: str, timeout: float, what: str) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.evaluate(expression):
                return
            time.sleep(0.25)
        raise RuntimeError(f"timed out after {timeout:.0f}s waiting for {what}")

    def close(self) -> None:
        self._socket.close()


def png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return struct.unpack(">II", data[16:24])


def capture(theme: str, timeout: float) -> bytes:
    server = _serve_docs(DOCS_DIR, _free_port(), None)
    devtools_port = _free_port()
    profile = tempfile.mkdtemp(prefix="carmine-og-")
    chrome = subprocess.Popen(
        [
            _find_chrome(),
            "--headless",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            f"--user-data-dir={profile}",
            f"--remote-debugging-port={devtools_port}",
            "--no-first-run",
            "--no-default-browser-check",
            "--hide-scrollbars",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    page = None
    try:
        deadline = time.monotonic() + 30
        tab = None
        while tab is None and time.monotonic() < deadline:
            tab = _page_tab(devtools_port)
            time.sleep(0.25)
        if tab is None:
            raise RuntimeError("Chrome's DevTools endpoint never came up")
        page = _Page(tab["webSocketDebuggerUrl"])
        page.call(
            "Emulation.setDeviceMetricsOverride",
            width=CAPTURE_WIDTH,
            height=CAPTURE_HEIGHT,
            deviceScaleFactor=1,
            mobile=False,
        )
        # The page follows the device's light or dark setting; the preview
        # is taken in the one asked for, the dark theme unless told otherwise.
        page.call(
            "Emulation.setEmulatedMedia",
            features=[{"name": "prefers-color-scheme", "value": theme}],
        )
        page.call("Page.navigate", url=f"http://127.0.0.1:{server.server_address[1]}/index.html")
        page.wait_for(
            "document.readyState === 'complete' && !!document.querySelector('.stage canvas')",
            timeout,
            "the mirror to mount",
        )
        page.wait_for(SAMPLE_READY, timeout, "the sample portrait to render")
        # The page's film grain is noise on purpose, and noise is the one
        # thing a PNG cannot compress: the preview is taken without it.
        page.evaluate(
            "document.head.append(Object.assign(document.createElement('style'),"
            " {textContent: 'body::after { display: none !important; }'}))"
        )
        # Picking a look blends into it; give the blend time to finish.
        page.evaluate(CLICK_PRESET.format(label="Velvet"))
        time.sleep(1.5)
        page.evaluate("new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))")
        time.sleep(0.5)
        shot = page.call("Page.captureScreenshot", format="png")
        return base64.b64decode(shot["data"])
    finally:
        if page is not None:
            page.close()
        chrome.terminate()
        try:
            chrome.wait(timeout=5)
        except subprocess.TimeoutExpired:
            chrome.kill()
        server.shutdown()
        shutil.rmtree(profile, ignore_errors=True)


def to_preview(capture_png: bytes) -> bytes:
    """Scales the capture to 1200x630 and fits it under the size limit."""
    from PIL import Image

    image = Image.open(io.BytesIO(capture_png)).convert("RGB")
    image = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    if out.tell() <= MAX_BYTES:
        return out.getvalue()
    palette = image.quantize(
        colors=256, method=Image.Quantize.FASTOCTREE, kmeans=2, dither=Image.Dither.FLOYDSTEINBERG
    )
    out = io.BytesIO()
    palette.save(out, format="PNG", optimize=True)
    return out.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="where to write the PNG")
    parser.add_argument("--theme", choices=("light", "dark"), default="dark")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per wait")
    args = parser.parse_args()

    if not (DOCS_DIR / "index.html").exists():
        print(f"{DOCS_DIR} has no index.html; run the web build first", file=sys.stderr)
        return 1
    data = to_preview(capture(args.theme, args.timeout))
    size = png_size(data)
    if size != (WIDTH, HEIGHT):
        print(f"captured {size[0]}x{size[1]}, expected {WIDTH}x{HEIGHT}", file=sys.stderr)
        return 1
    if len(data) > MAX_BYTES:
        print(f"captured {len(data) // 1024} KB, over the {MAX_BYTES // 1024} KB limit", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"wrote {args.out} ({size[0]}x{size[1]}, {len(data) // 1024} KB, {args.theme} theme)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
