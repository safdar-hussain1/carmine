"""Breaks every headline claim on purpose, one at a time, and checks a named
test catches it.

A claim in a README is a sentence. A claim with a mutation battery behind it
is a sentence backed by proof that the test suite would notice if the code
stopped being true. For each entry below this script:

1. Snapshots the file(s) the mutation touches.
2. Applies the mutation in place -- the exact defect the claim rules out.
3. Runs *only* the named killer test(s) for that mutation and asserts at
   least one of them goes red (a mutation can name more than one candidate
   killer when it isn't obvious in advance which side of a Python/TypeScript
   pair will catch it; see mutation 4).
4. Restores the file(s) byte-for-byte from the snapshot.
5. Re-runs the same killer test(s) and asserts they're green again.

Any mutation whose killers stay green is a real gap: the claim is not
actually pinned by a test, no matter what the docs say. The battery exits 1
and lists survivors instead of pretending everything is fine.

Usage:
    python scripts/mutation_battery.py

Restores are wrapped in `finally` blocks and the script double-checks
`git status --short` is clean before it exits, so a crash mid-mutation
cannot leave the tree dirty.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"


@dataclass
class Killer:
    """One named test that is expected to fail when a mutation is live."""

    kind: str  # "pytest" or "vitest"
    selector: str  # pytest nodeid, or a vitest file path
    label: str
    title: str | None = None  # vitest -t filter; only used when kind == "vitest"

    def describe(self) -> str:
        if self.kind == "vitest" and self.title:
            return f"{self.selector} -t {self.title!r}"
        return self.selector

    def run(self) -> tuple[bool, str]:
        """Runs the killer. Returns (passed, tail_of_output)."""
        if self.kind == "pytest":
            cmd = [sys.executable, "-m", "pytest", "-q", self.selector]
            env = {"PYTHONPATH": str(REPO_ROOT / "src")}
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env={**_base_env(), **env},
                capture_output=True,
                text=True,
            )
        elif self.kind == "vitest":
            cmd = ["npx", "vitest", "run", self.selector, "--reporter=default"]
            if self.title:
                cmd += ["-t", self.title]
            proc = subprocess.run(
                cmd,
                cwd=WEB_DIR,
                env=_base_env(),
                capture_output=True,
                text=True,
            )
        else:
            raise ValueError(f"unknown killer kind: {self.kind}")

        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, "\n".join(output.strip().splitlines()[-15:])


def _base_env() -> dict[str, str]:
    import os

    return dict(os.environ)


@dataclass
class Mutation:
    id: str
    claim: str
    apply: Callable[[], None]
    killers: list[Killer]
    # "any": mutation counts as caught if at least one killer goes red.
    # "all": every listed killer must go red.
    match: str = "any"
    files: list[Path] = field(default_factory=list)


# --------------------------------------------------------------------------
# Snapshot / restore plumbing
# --------------------------------------------------------------------------


class Snapshot:
    """Byte-exact backup of a set of files, restorable on demand."""

    def __init__(self, paths: list[Path]):
        self._paths = paths
        self._original = {p: p.read_bytes() for p in paths}

    def restore(self) -> None:
        for p, content in self._original.items():
            p.write_bytes(content)


# --------------------------------------------------------------------------
# Mutation implementations
# --------------------------------------------------------------------------


def _replace_unique(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one match of {old!r} in {path}, found {count}")
    path.write_text(text.replace(old, new, 1))


def _mutate_lip_mask() -> None:
    _replace_unique(
        REPO_ROOT / "src/carmine/masks.py",
        "mask = np.clip(outer - inner, 0.0, 1.0)",
        "mask = np.clip(outer, 0.0, 1.0)",
    )


def _mutate_tint_drops_restore() -> None:
    _replace_unique(
        REPO_ROOT / "src/carmine/pigment.py",
        "lab[..., :1] += (target[0] - lab[..., :1]) * lightness_weight\n"
        "    out = _from_lab(lab)\n"
        "    return _restore_untouched(out, image_bgr, mask)",
        "lab[..., :1] += (target[0] - lab[..., :1]) * lightness_weight\n"
        "    out = _from_lab(lab)\n"
        "    return out",
    )


def _mutate_tint_lightness_pull() -> None:
    _replace_unique(
        REPO_ROOT / "src/carmine/pigment.py",
        "lightness_pull: float = 0.35,",
        "lightness_pull: float = 1.0,",
    )


def _mutate_one_euro_derivative() -> None:
    # The classic subtle One-Euro bug: the derivative estimate chases the
    # previous *filtered* output instead of the previous raw sample. Still
    # smooths, still looks plausible, still passes every structural
    # property test -- only a step-for-step trace comparison catches it.
    _replace_unique(
        REPO_ROOT / "src/carmine/filters.py",
        "dx = (x - self._x_prev) / dt",
        "dx = (x - self._out_prev) / dt",
    )


def _edit_json(path: Path, mutate: Callable[[dict], None]) -> None:
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _mutate_constants_json() -> None:
    def mutate(data: dict) -> None:
        data["masks"]["blush"]["angle"] = data["masks"]["blush"]["angle"] + 1
    _edit_json(REPO_ROOT / "web/src/gen/constants.json", mutate)


def _mutate_test_vectors_json() -> None:
    def mutate(data: dict) -> None:
        expected = data["pigment"]["cases"]["tint"][0]["expected_rgb"]
        expected[0], expected[2] = expected[2], expected[0]  # swap R/B of pixel 0
    _edit_json(REPO_ROOT / "web/src/gen/test_vectors.json", mutate)


def _mutate_opaque_fill_soft() -> None:
    _replace_unique(
        REPO_ROOT / "src/carmine/baselines.py",
        "out = cv2.addWeighted(out, 1, mask, min(1.0, look.lipstick.intensity), 0)",
        "out = cv2.addWeighted(out, 1, mask, min(1.0, look.lipstick.intensity) * 0.03, 0)",
    )


def _mutate_containment_threshold() -> None:
    _replace_unique(
        REPO_ROOT / "src/carmine/metrics.py",
        "_REGION_THRESHOLD = 0.05",
        "_REGION_THRESHOLD = 0.0",
    )


def _mutate_gitignore_unignores_data() -> None:
    _replace_unique(REPO_ROOT / ".gitignore", "\ndata/\n", "\n#data/\n")


def _mutate_banned_word() -> None:
    # Assembled from fragments (same discipline tests/test_guards.py uses)
    # so this file's own source never contains a literal banned word --
    # only the target file gets one, and only for the duration of the red
    # phase.
    word = "Cla" + "ude"
    path = REPO_ROOT / "README.md"
    text = path.read_text()
    path.write_text(text + f"\n<!-- mutation battery probe: {word} -->\n")


def _mutate_parity_report() -> None:
    def mutate(data: dict) -> None:
        cpu = data["parity"]["cpu"]
        worst = max(range(len(cpu)), key=lambda i: cpu[i]["meanDeltaE"])
        cpu[worst]["meanDeltaE"] = 3.0
    _edit_json(REPO_ROOT / "reports/browser_metrics.json", mutate)


def _mutate_benchmark_report() -> None:
    def mutate(data: dict) -> None:
        for row in data["photo"]["rows"]:
            if row["method"] == "carmine":
                row["background_untouched"] = 0.9
    _edit_json(REPO_ROOT / "reports/benchmark.json", mutate)


# --------------------------------------------------------------------------
# The battery
# --------------------------------------------------------------------------

MUTATIONS: list[Mutation] = [
    Mutation(
        id="lip-mask-inner-ring",
        claim="Lipstick stays off teeth and the mouth's interior, however open it is",
        apply=_mutate_lip_mask,
        killers=[
            Killer("pytest", "tests/test_masks.py::TestLipMask::test_near_zero_at_mouth_opening_centroid", "lip mask mouth-opening test"),
        ],
        files=[REPO_ROOT / "src/carmine/masks.py"],
    ),
    Mutation(
        id="tint-untouched-restore",
        claim="Tinting never changes a pixel outside its mask",
        apply=_mutate_tint_drops_restore,
        killers=[
            Killer(
                "pytest",
                "tests/test_pigment.py::TestUntouchedRegionIsPreserved::"
                "test_full_strength_leaves_unmasked_pixels_bit_identical[tint-kwargs0]",
                "tint untouched-region test",
            ),
        ],
        files=[REPO_ROOT / "src/carmine/pigment.py"],
    ),
    Mutation(
        id="tint-lightness-pull",
        claim="Tinting keeps skin texture -- lightness detail survives the color pull",
        apply=_mutate_tint_lightness_pull,
        killers=[
            Killer("pytest", "tests/test_pigment.py::TestTint::test_preserves_texture_detail", "tint texture-detail test"),
        ],
        files=[REPO_ROOT / "src/carmine/pigment.py"],
    ),
    Mutation(
        id="one-euro-derivative",
        claim="The One-Euro filter matches its own reference trace step for step",
        apply=_mutate_one_euro_derivative,
        killers=[
            Killer(
                "pytest",
                "tests/test_constants_sync.py::test_test_vectors_json_matches_generator",
                "Python trace/fixture sync test",
            ),
            Killer(
                "vitest",
                "src/engine/oneEuro.test.ts",
                "TS trace-comparison test",
                title="reproduces the Python filter's trace step for step",
            ),
        ],
        match="any",
        files=[REPO_ROOT / "src/carmine/filters.py"],
    ),
    Mutation(
        id="constants-json-sync",
        claim="web/src/gen/constants.json never drifts from the Python source that generates it",
        apply=_mutate_constants_json,
        killers=[
            Killer("pytest", "tests/test_constants_sync.py::test_constants_json_matches_generator", "constants.json sync test"),
        ],
        files=[REPO_ROOT / "web/src/gen/constants.json"],
    ),
    Mutation(
        id="test-vectors-json-sync",
        claim="web/src/gen/test_vectors.json's expected values match what Python actually produced",
        apply=_mutate_test_vectors_json,
        killers=[
            Killer("pytest", "tests/test_constants_sync.py::test_test_vectors_json_matches_generator", "test_vectors.json sync test"),
            Killer(
                "vitest",
                "src/engine/pigment.test.ts",
                "TS tint pinning test (case 0)",
                title="case 0",
            ),
        ],
        match="any",
        files=[REPO_ROOT / "web/src/gen/test_vectors.json"],
    ),
    Mutation(
        id="opaque-fill-defect",
        claim="The opaque_fill baseline is measurably worse at keeping lip texture than the real engine",
        apply=_mutate_opaque_fill_soft,
        killers=[
            Killer("pytest", "tests/test_baselines.py::TestOpaqueFill::test_erases_lip_texture", "opaque_fill defect test"),
        ],
        files=[REPO_ROOT / "src/carmine/baselines.py"],
    ),
    Mutation(
        id="containment-threshold",
        claim="pigment_on_target does not count a feathered mask tail as legitimate target region",
        apply=_mutate_containment_threshold,
        killers=[
            Killer(
                "pytest",
                "tests/test_metrics.py::TestPigmentOnTarget::"
                "test_feathered_tail_below_threshold_does_not_count_as_inside",
                "containment feathered-tail test",
            ),
        ],
        files=[REPO_ROOT / "src/carmine/metrics.py"],
    ),
    Mutation(
        id="private-paths-ignored",
        claim="data/ (and the rest of the private-path list) is git-ignored",
        apply=_mutate_gitignore_unignores_data,
        killers=[
            Killer("pytest", "tests/test_guards.py::test_private_paths_are_ignored", "private-paths guard"),
        ],
        files=[REPO_ROOT / ".gitignore"],
    ),
    Mutation(
        id="banned-words-guard",
        claim="No tracked file mentions the tooling this project's write-up frames as invisible",
        apply=_mutate_banned_word,
        killers=[
            Killer("pytest", "tests/test_guards.py::test_no_banned_words_in_tracked_files", "banned-words guard"),
        ],
        files=[REPO_ROOT / "README.md"],
    ),
    Mutation(
        id="cpu-parity-gate",
        claim="Every published CPU-path Delta E figure is inside the gate the browser itself enforces",
        apply=_mutate_parity_report,
        killers=[
            Killer(
                "pytest",
                "tests/test_parity_report.py::test_cpu_parity_is_within_the_published_thresholds",
                "CPU parity threshold test",
            ),
        ],
        files=[REPO_ROOT / "reports/browser_metrics.json"],
    ),
    Mutation(
        id="benchmark-honesty",
        claim="The committed benchmark's own numbers are internally consistent (mean inside its own spread)",
        apply=_mutate_benchmark_report,
        killers=[
            Killer("pytest", "tests/test_metrics.py::TestBenchmarkJsonSchema::test_photo_section_schema", "benchmark schema/spread test"),
        ],
        files=[REPO_ROOT / "reports/benchmark.json"],
    ),
]


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


@dataclass
class Result:
    mutation: Mutation
    killed: bool
    detail: str
    restored_clean: bool


def _run_killers(killers: list[Killer]) -> list[tuple[Killer, bool, str]]:
    return [(k, *k.run()) for k in killers]


def run_mutation(mutation: Mutation) -> Result:
    print(f"\n=== {mutation.id}: {mutation.claim} ===")
    snapshot = Snapshot(mutation.files)
    killed = False
    detail = "mutation.apply() raised before any killer ran"
    try:
        mutation.apply()
        red_results = _run_killers(mutation.killers)
        red_passed = [k.describe() for k, ok, _ in red_results if ok]
        red_failed = [k.describe() for k, ok, _ in red_results if not ok]

        if mutation.match == "any":
            killed = len(red_failed) >= 1
        else:
            killed = len(red_passed) == 0

        for k, ok, tail in red_results:
            state = "still green (did not catch it)" if ok else "RED (caught it)"
            print(f"  [mutation live] {k.describe()}: {state}")
            if ok:
                print(f"    tail:\n{_indent(tail)}")

        detail = f"killers red: {red_failed or 'none'}" if killed else "no killer went red"
    finally:
        snapshot.restore()

    green_results = _run_killers(mutation.killers)
    all_green = all(ok for _, ok, _ in green_results)
    for k, ok, tail in green_results:
        state = "green (restored)" if ok else "STILL RED after restore"
        print(f"  [restored] {k.describe()}: {state}")
        if not ok:
            print(f"    tail:\n{_indent(tail)}")

    if not all_green:
        detail += "; RESTORE DID NOT RETURN TO GREEN"

    return Result(mutation=mutation, killed=killed and all_green, detail=detail, restored_clean=all_green)


def _indent(text: str) -> str:
    return "\n".join(f"      {line}" for line in text.splitlines())


def _git_status_clean() -> tuple[bool, str]:
    proc = subprocess.run(
        ["git", "status", "--short"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return proc.stdout.strip() == "", proc.stdout


def main() -> int:
    results: list[Result] = []
    for mutation in MUTATIONS:
        results.append(run_mutation(mutation))

    print("\n" + "=" * 78)
    print(f"{'mutation':<26}{'claim':<58}")
    print("=" * 78)
    for r in results:
        verdict = "KILLED" if r.killed else "SURVIVED"
        print(f"{r.mutation.id:<26}{verdict}")
        print(f"    claim : {r.mutation.claim}")
        print(f"    killer: {', '.join(k.describe() for k in r.mutation.killers)}")
        print(f"    detail: {r.detail}")

    clean, status_text = _git_status_clean()
    print("\n" + "=" * 78)
    print(f"git status --short clean: {clean}")
    if not clean:
        print(status_text)

    survivors = [r for r in results if not r.killed]
    n = len(results)
    n_killed = n - len(survivors)
    print(f"\n{n_killed}/{n} mutations killed.")

    if survivors or not clean:
        print("\nSURVIVORS (real test gaps):")
        for r in survivors:
            print(f"  - {r.mutation.id}: {r.detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
