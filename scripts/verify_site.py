"""Headlessly verifies that the built site's selftest passes.

Serves `docs/` over HTTP, launches headless Chrome with software rendering
(SwiftShader, since no GPU is available in CI/dev containers), navigates to
`?selftest=1`, and polls the tab's title over the DevTools protocol until it
starts with "SELFTEST" (see web/src/lib/selftest.ts). Exits 0 and prints the
title on PASS; exits 1 and prints diagnostics otherwise.

Usage:
    python scripts/verify_site.py [--url URL] [--timeout SECONDS]

Later tasks add more selftest checks to the same page; this script doesn't
need to change to pick those up, since it only watches the aggregate title.
"""

from __future__ import annotations

import argparse
import http.server
import json
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
    "chromium-browser",
]


def _find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise RuntimeError(f"no Chrome/Chromium binary found among {CHROME_CANDIDATES}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve_docs(directory: Path, port: int) -> http.server.ThreadingHTTPServer:
    handler_cls = type(
        "DocsHandler",
        (http.server.SimpleHTTPRequestHandler,),
        {},
    )

    def make_handler(*args, **kwargs):
        return handler_cls(*args, directory=str(directory), **kwargs)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), make_handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _get_tab_title(devtools_port: int) -> str | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{devtools_port}/json", timeout=2) as resp:
            tabs = json.loads(resp.read())
    except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
        return None
    for tab in tabs:
        if tab.get("type") == "page":
            return tab.get("title")
    return None


def verify(url: str | None, timeout: float) -> tuple[bool, str]:
    """Runs the headless-Chrome selftest verification.

    Returns (passed, title_or_diagnostic).
    """
    server = None
    chrome_proc = None
    try:
        target_url = url
        if target_url is None:
            if not DOCS_DIR.exists():
                return False, f"docs/ not found at {DOCS_DIR}; run `npm run build` first"
            srv_port = _free_port()
            server = _serve_docs(DOCS_DIR, srv_port)
            target_url = f"http://127.0.0.1:{srv_port}/index.html?selftest=1"

        chrome_bin = _find_chrome()
        devtools_port = _free_port()
        chrome_proc = subprocess.Popen(
            [
                chrome_bin,
                "--headless",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                f"--remote-debugging-port={devtools_port}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-gpu-sandbox",
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                target_url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.monotonic() + timeout
        title = None
        while time.monotonic() < deadline:
            title = _get_tab_title(devtools_port)
            if title and title.startswith("SELFTEST"):
                break
            time.sleep(0.5)

        if title is None:
            return False, f"timed out after {timeout}s waiting for Chrome DevTools to respond"
        if not title.startswith("SELFTEST"):
            return False, f"timed out after {timeout}s; last title was: {title!r}"

        return title.startswith("SELFTEST PASS"), title
    finally:
        if chrome_proc is not None:
            chrome_proc.terminate()
            try:
                chrome_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                chrome_proc.kill()
        if server is not None:
            server.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=None,
        help="Override the URL to load (default: serve docs/ locally with ?selftest=1 appended)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the selftest title (default: 60)",
    )
    args = parser.parse_args()

    passed, title = verify(args.url, args.timeout)
    print(title)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
