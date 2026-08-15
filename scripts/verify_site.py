"""Headlessly verifies that the built site's selftest passes.

Serves `docs/` over HTTP, launches headless Chrome with software rendering
(SwiftShader, since no GPU is available in CI/dev containers), navigates to
`?selftest=1`, and polls the tab's title over the DevTools protocol until it
starts with "SELFTEST" (see web/src/lib/selftest.ts). Exits 0 and prints the
title on PASS; exits 1 and prints diagnostics otherwise.

Usage:
    python scripts/verify_site.py [--url URL] [--timeout SECONDS]
                                  [--expect-checks N] [--with-parity]

The title carries the aggregate result and the number of checks that ran, so
adding a check needs no change here -- but `--expect-checks` is enforced by
default, because a check that silently stops being registered would otherwise
still report PASS.

`--with-parity` additionally mounts `reports/parity_fixtures/` at `/parity/`
and requires the parity checks to have actually run. Those fixtures are
renders of dataset faces: they are git-ignored, they are never copied into
`docs/`, and this mount is the only way they are ever served -- over
localhost, to a headless browser, for the duration of one verification. The
default (no flag) run is the deployed-site path, where `/parity/` does not
exist, the checks skip themselves, and the selftest still passes.

With `--with-parity`, `window.__carmine_results` is pulled back over the
DevTools protocol and the parity and timing numbers are written to
`reports/browser_metrics.json`. That file holds numbers only -- no image
data -- which is why it is the part of this measurement that gets committed.
The write merges into whatever is already there, because `--timing-only`
contributes a `timing_hardware` block this run cannot produce (it needs
Chrome launched without software rasterization).

Regenerating the metrics file from a clean slate therefore means running
`--with-parity` first and `--timing-only` second; running them the other way
round leaves a `timing_hardware` block measured against the previous bundle.
"""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
PARITY_DIR = REPO_ROOT / "reports" / "parity_fixtures"
METRICS_PATH = REPO_ROOT / "reports" / "browser_metrics.json"

# Number of checks web/src/main.ts registers. Bumped whenever one is added.
EXPECTED_CHECKS = 9

# Checks that only run when the parity fixtures are mounted.
PARITY_CHECKS = ("parity-cpu", "parity-gpu")

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


def _serve_docs(
    directory: Path, port: int, parity_dir: Path | None
) -> http.server.ThreadingHTTPServer:
    """Serve `directory` at /, optionally overlaying `parity_dir` at /parity/.

    Two roots rather than a copy: copying the fixtures into `docs/` would put
    dataset faces one `npm run build` away from being published, and the
    whole point is that they never enter the shipped tree.
    """

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def translate_path(self, path):
            if parity_dir is not None:
                # Strip the query/fragment the base class also discards
                # before deciding whether this is a parity request.
                clean = path.split("?", 1)[0].split("#", 1)[0]
                if clean.startswith("/parity/"):
                    relative = clean[len("/parity/") :]
                    # posixpath-style containment check: a "..' segment must
                    # not be able to escape the fixture directory.
                    resolved = (parity_dir / relative).resolve()
                    if parity_dir.resolve() in resolved.parents or resolved == parity_dir.resolve():
                        return str(resolved)
                    return str(parity_dir / "__forbidden__")
            return super().translate_path(path)

        def log_message(self, *args):  # noqa: D102 - silence request logging
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _tabs(devtools_port: int) -> list[dict] | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{devtools_port}/json", timeout=2) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
        return None


def _page_tab(devtools_port: int) -> dict | None:
    tabs = _tabs(devtools_port)
    if tabs is None:
        return None
    for tab in tabs:
        if tab.get("type") == "page":
            return tab
    return None


def _get_tab_title(devtools_port: int) -> str | None:
    tab = _page_tab(devtools_port)
    return tab.get("title") if tab else None


class _DevToolsSocket:
    """The smallest WebSocket client that can drive one DevTools command.

    The DevTools protocol needs a WebSocket, and this project's Python
    dependencies are the engine's -- OpenCV, MediaPipe, NumPy. Pulling in a
    WebSocket library so a verification script can read one JSON blob would
    put a dependency in the install path of everyone who only wants the
    engine, so the ~60 lines of framing it actually needs live here instead.

    Only what CDP over loopback requires is implemented: a client handshake,
    masked text frames out, and reassembly of text/continuation frames in.
    Ping and close are handled because Chrome sends them; binary frames and
    compression extensions are not, because Chrome never offers them here.
    """

    def __init__(self, url: str, timeout: float) -> None:
        parsed = urllib.parse.urlparse(url)
        self._socket = socket.create_connection(
            (parsed.hostname, parsed.port or 80), timeout=timeout
        )
        self._buffer = b""
        key = base64.b64encode(os.urandom(16)).decode()
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{parsed.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self._socket.sendall(request.encode())
        header = self._read_until(b"\r\n\r\n")
        if b" 101 " not in header.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"DevTools refused the WebSocket upgrade: {header[:120]!r}")

    def _read_until(self, marker: bytes) -> bytes:
        while marker not in self._buffer:
            chunk = self._socket.recv(65536)
            if not chunk:
                raise RuntimeError("DevTools closed the connection during the handshake")
            self._buffer += chunk
        head, self._buffer = self._buffer.split(marker, 1)
        return head + marker

    def _read_exactly(self, count: int) -> bytes:
        while len(self._buffer) < count:
            chunk = self._socket.recv(65536)
            if not chunk:
                raise RuntimeError("DevTools closed the connection mid-frame")
            self._buffer += chunk
        data, self._buffer = self._buffer[:count], self._buffer[count:]
        return data

    def send(self, payload: str) -> None:
        data = payload.encode()
        length = len(data)
        header = bytearray([0x81])  # FIN + text opcode
        if length < 126:
            header.append(0x80 | length)
        elif length < (1 << 16):
            header.append(0x80 | 126)
            header += length.to_bytes(2, "big")
        else:
            header.append(0x80 | 127)
            header += length.to_bytes(8, "big")
        mask = os.urandom(4)
        header += mask
        # Client frames must be masked (RFC 6455 §5.3).
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(data))
        self._socket.sendall(bytes(header) + masked)

    def _recv_frame(self) -> tuple[int, bool, bytes]:
        first, second = self._read_exactly(2)
        opcode = first & 0x0F
        fin = bool(first & 0x80)
        length = second & 0x7F
        if length == 126:
            length = int.from_bytes(self._read_exactly(2), "big")
        elif length == 127:
            length = int.from_bytes(self._read_exactly(8), "big")
        if second & 0x80:  # server frames are never masked, but be safe
            mask = self._read_exactly(4)
            body = self._read_exactly(length)
            body = bytes(byte ^ mask[i % 4] for i, byte in enumerate(body))
        else:
            body = self._read_exactly(length)
        return opcode, fin, body

    def recv(self) -> str:
        parts: list[bytes] = []
        while True:
            opcode, fin, body = self._recv_frame()
            if opcode == 0x8:
                raise RuntimeError("DevTools closed the WebSocket")
            if opcode == 0x9:  # ping -> pong, unmasked payload echoed back
                self.send_pong(body)
                continue
            if opcode == 0xA:
                continue
            parts.append(body)
            if fin:
                return b"".join(parts).decode()

    def send_pong(self, payload: bytes) -> None:
        mask = os.urandom(4)
        header = bytes([0x8A, 0x80 | len(payload)]) + mask
        self._socket.sendall(
            header + bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        )

    def close(self) -> None:
        try:
            self._socket.close()
        except OSError:
            pass


def _evaluate(ws_url: str, expression: str, timeout: float = 60.0):
    """Run `expression` in the page and return its value.

    The expression is evaluated with `returnByValue`, so what comes back is
    already JSON rather than a remote object handle.
    """
    connection = _DevToolsSocket(ws_url, timeout)
    try:
        connection.send(
            json.dumps(
                {
                    "id": 1,
                    "method": "Runtime.evaluate",
                    "params": {
                        "expression": expression,
                        "returnByValue": True,
                        "awaitPromise": True,
                    },
                }
            )
        )
        while True:
            message = json.loads(connection.recv())
            if message.get("id") != 1:
                # Unsolicited protocol events (console messages, lifecycle
                # notifications) share the socket; skip anything not ours.
                continue
            result = message.get("result", {})
            if "exceptionDetails" in result:
                raise RuntimeError(f"page evaluation failed: {result['exceptionDetails']}")
            return result.get("result", {}).get("value")
    finally:
        connection.close()


# Renderer substrings that mean Chrome fell back to software rasterization.
SOFTWARE_RENDERERS = ("swiftshader", "llvmpipe", "software", "lavapipe")


def is_software_renderer(renderer: str) -> bool:
    lowered = (renderer or "").lower()
    return any(name in lowered for name in SOFTWARE_RENDERERS)


def verify(
    url: str | None,
    timeout: float,
    with_parity: bool,
    software_gl: bool = True,
) -> tuple[bool, str, dict | None]:
    """Runs the headless-Chrome selftest verification.

    Returns (passed, title_or_diagnostic, results). `results` is the page's
    `window.__carmine_results`, pulled back when `with_parity` is set or when
    `software_gl` is off (the hardware-timing attempt wants the numbers even
    though it mounts no fixtures).

    `software_gl` forces SwiftShader, which is the right default: it is
    reproducible, it works on machines with no GPU at all, and every
    correctness check here is about what the code does rather than how fast
    it does it. Turning it off asks Chrome for whatever real driver it can
    find, which is only worth doing for the timing numbers -- and the caller
    has to check the renderer string afterwards, because Chrome will happily
    fall back to software without saying so.
    """
    server = None
    chrome_proc = None
    try:
        target_url = url
        parity_dir = None
        if with_parity:
            if not PARITY_DIR.is_dir():
                return (
                    False,
                    f"parity fixtures not found at {PARITY_DIR}; "
                    "run scripts/export_parity_fixtures.py first",
                    None,
                )
            parity_dir = PARITY_DIR

        if target_url is None:
            if not DOCS_DIR.exists():
                return False, f"docs/ not found at {DOCS_DIR}; run the web build first", None
            srv_port = _free_port()
            server = _serve_docs(DOCS_DIR, srv_port, parity_dir)
            target_url = f"http://127.0.0.1:{srv_port}/index.html?selftest=1"
        elif parity_dir is not None:
            return False, "--with-parity cannot be combined with --url", None

        chrome_bin = _find_chrome()
        devtools_port = _free_port()
        gl_flags = (
            ["--headless", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
            if software_gl
            # --headless=new runs the full browser compositor rather than the
            # old headless shell, which is the only mode with any chance of
            # reaching a real driver.
            else ["--headless=new", "--enable-gpu", "--ignore-gpu-blocklist"]
        )
        chrome_proc = subprocess.Popen(
            [
                chrome_bin,
                *gl_flags,
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
            return False, f"timed out after {timeout}s waiting for Chrome DevTools", None
        if not title.startswith("SELFTEST"):
            return False, f"timed out after {timeout}s; last title was: {title!r}", None

        results = None
        if with_parity or not software_gl:
            tab = _page_tab(devtools_port)
            ws_url = tab.get("webSocketDebuggerUrl") if tab else None
            if not ws_url:
                return False, "no DevTools WebSocket URL for the page", None
            results = _evaluate(ws_url, "JSON.stringify(window.__carmine_results)")
            results = json.loads(results) if isinstance(results, str) else results

        return title.startswith("SELFTEST PASS"), title, results
    finally:
        if chrome_proc is not None:
            chrome_proc.terminate()
            try:
                chrome_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                chrome_proc.kill()
        if server is not None:
            server.shutdown()


def _format_number(value: float) -> str:
    return f"{value:.3f}"


def summarize(results: dict) -> str:
    """A human-readable table of the parity and timing numbers."""
    lines: list[str] = []
    parity = results.get("parity") or {}
    for label, key in (("cpu (fixed landmarks)", "cpu"), ("gpu (live shader)", "gpu"),
                       ("end-to-end (own landmarker)", "endToEnd")):
        cases = parity.get(key)
        if not cases:
            continue
        lines.append(f"\n{label}:")
        lines.append(f"  {'frame':<16} {'look':<14} {'mean ΔE':>9} {'p99 ΔE':>9} {'max ΔE':>9} {'outside':>8}")
        for case in cases:
            lines.append(
                f"  {case['frame']:<16} {case['look']:<14} "
                f"{_format_number(case['meanDeltaE']):>9} "
                f"{_format_number(case['p99DeltaE']):>9} "
                f"{_format_number(case['maxDeltaE']):>9} "
                f"{case['changedOutsideSupport']:>8}"
            )
    if parity.get("glRenderer"):
        lines.append(f"\nGL renderer: {parity['glRenderer']}")

    timing = results.get("timing")
    if timing:
        lines.append(
            f"\ntiming ({timing['width']}x{timing['height']}, {timing['frames']} frames, "
            f"look={timing['look']}, interocular={_format_number(timing['interocularPx'])}px):"
        )
        lines.append(f"  {'stage':<24} {'median ms':>10} {'min':>9} {'max':>9}")
        for stage in ("detect", "buildMasks", "draw"):
            entry = timing[stage]
            label = stage if stage == "detect" else f"{stage} (reference)"
            lines.append(
                f"  {label:<24} {_format_number(entry['median']):>10} "
                f"{_format_number(entry['min']):>9} {_format_number(entry['max']):>9}"
            )
        lines.append(
            f"  {'total (reference)':<24} {_format_number(timing['totalMedian']):>10}"
            f"   masks {timing['maskWidth']}x{timing['maskHeight']}"
        )
        live = timing.get("livePath")
        if live:
            for stage in ("buildMasks", "draw"):
                entry = live[stage]
                lines.append(
                    f"  {stage + ' (live)':<24} {_format_number(entry['median']):>10} "
                    f"{_format_number(entry['min']):>9} {_format_number(entry['max']):>9}"
                )
            lines.append(
                f"  {'total (live)':<24} {_format_number(live['totalMedian']):>10}"
                f"   masks {live['maskWidth']}x{live['maskHeight']}"
            )
            reference = timing["buildMasks"]["median"]
            if live["buildMasks"]["median"] > 0:
                speedup = reference / live["buildMasks"]["median"]
                lines.append(f"  mask stage speedup: {speedup:.1f}x")
        lines.append(f"  renderer: {timing['glRenderer']}")
    return "\n".join(lines)


def run_hardware_timing(args) -> int:
    """Attempts a timing run on a real GPU and records whatever happens.

    Every other number this script produces comes from SwiftShader, which is
    reproducible but is not the hardware anyone runs the site on. This asks
    Chrome for a real driver instead and, if it gets one, merges the timing
    block into the metrics file under `timing_hardware`.

    It records the *attempt* either way. A run that silently fell back to
    software and got published as a GPU number would be worse than no GPU
    number at all, so the renderer string decides, and a fallback is written
    down as a fallback rather than dropped on the floor.
    """
    if not args.metrics_out.exists():
        print(f"{args.metrics_out} not found; run --with-parity first")
        return 1

    timeout = args.timeout if args.timeout != 120.0 else 900.0
    passed, title, results = verify(args.url, timeout, with_parity=False, software_gl=False)
    print(title)

    payload = json.loads(args.metrics_out.read_text(encoding="utf-8"))
    timing = (results or {}).get("timing") if passed else None
    renderer = (timing or {}).get("glRenderer", "")

    if not passed:
        record = {"attempted": True, "recorded": False, "reason": f"run failed: {title}"}
    elif not timing:
        record = {"attempted": True, "recorded": False, "reason": "page reported no timing block"}
    elif is_software_renderer(renderer):
        record = {
            "attempted": True,
            "recorded": False,
            "reason": f"Chrome fell back to software rasterization: {renderer}",
            "glRenderer": renderer,
        }
    else:
        record = {"attempted": True, "recorded": True, "glRenderer": renderer, "timing": timing}

    payload["timing_hardware"] = record
    args.metrics_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if record["recorded"]:
        print(f"\nhardware timing recorded on: {renderer}")
        print(summarize({"timing": timing}))
    else:
        print(f"\nhardware timing not recorded: {record['reason']}")
    print(f"\nupdated {args.metrics_out}")
    # Not reaching a real GPU is a fact about the machine, not a failure of
    # the build; the outcome is recorded either way.
    return 0


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
        default=120.0,
        help="Seconds to wait for the selftest title (default: 120)",
    )
    parser.add_argument(
        "--expect-checks",
        type=int,
        default=EXPECTED_CHECKS,
        help=f"Fail unless exactly this many checks ran (default: {EXPECTED_CHECKS}; 0 disables)",
    )
    parser.add_argument(
        "--with-parity",
        action="store_true",
        help="Mount reports/parity_fixtures at /parity/, require the parity checks to run, "
        "and write reports/browser_metrics.json",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=METRICS_PATH,
        help=f"Where --with-parity writes its numbers (default: {METRICS_PATH.name})",
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="Re-run without forcing SwiftShader and merge the timing numbers into the "
        "metrics file as timing_hardware, if Chrome reaches a real GPU",
    )
    args = parser.parse_args()

    if args.timing_only:
        return run_hardware_timing(args)

    # The parity and timing checks are far heavier than the six structural
    # ones (six full-frame CPU renders plus 120 landmark detections on a
    # software rasterizer), so they get a longer default budget.
    timeout = args.timeout
    if args.with_parity and timeout == 120.0:
        timeout = 900.0

    passed, title, results = verify(args.url, timeout, args.with_parity)
    print(title)
    if not passed:
        # Print whatever was measured before the failing check threw: a
        # parity check that fails its threshold has already recorded the
        # per-fixture numbers, and those numbers are the diagnosis.
        if results:
            print(summarize(results))
        return 1

    if args.expect_checks:
        match = re.search(r"n=(\d+)", title)
        ran = int(match.group(1)) if match else -1
        if ran != args.expect_checks:
            print(f"expected {args.expect_checks} checks, {ran} ran")
            return 1

    if not args.with_parity:
        return 0

    if not results:
        print("no results pulled from the page")
        return 1

    selftest = results.get("selftest") or {}
    by_name = {entry["name"]: entry for entry in selftest.get("results", [])}
    for name in PARITY_CHECKS:
        entry = by_name.get(name)
        if entry is None:
            print(f"check {name} did not run")
            return 1
        if entry.get("skipped"):
            print(f"check {name} skipped ({entry.get('reason')}) but --with-parity requires it")
            return 1

    parity = results.get("parity") or {}
    if parity.get("skipped"):
        print(f"parity measurement skipped: {parity.get('reason')}")
        return 1

    # Merge rather than replace. `timing_hardware` is written by a separate
    # `--timing-only` run (it needs a different Chrome configuration, so it
    # cannot be produced here), and rewriting the file from scratch would
    # delete it every time this ran -- quietly, since nothing downstream
    # requires it to be present.
    payload = {}
    if args.metrics_out.exists():
        try:
            payload = json.loads(args.metrics_out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload["parity"] = parity
    payload["timing"] = results.get("timing")
    payload["selftest"] = {
        "count": selftest.get("count"),
        "skipped": selftest.get("skipped"),
        "pass": selftest.get("pass"),
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(summarize(results))
    print(f"\nwrote {args.metrics_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
