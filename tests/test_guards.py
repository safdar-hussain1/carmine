"""Structural guard tests.

These tests protect repository hygiene invariants: private/dataset paths stay
git-ignored, no leaked local file paths, no process/planning documents get
tracked, and no references to AI tooling end up in tracked text files.

Banned strings are assembled from concatenated fragments so this file never
contains a literal match of what it is guarding against.
"""

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
    ".ico", ".task", ".wasm", ".woff", ".woff2", ".ttf",
    ".otf", ".eot", ".pdf", ".mp4", ".zip", ".bz2", ".dat",
}

SKIP_PATHS = {
    "web/package-lock.json",
}

# A line in .gitignore that is *only* a dot-directory ignore pattern (e.g. a
# local tool workspace folder) is infrastructure that keeps such folders out
# of git, not prose mentioning any tool by name. Such lines are exempt from
# the banned-words scan; everything else in .gitignore (including comments)
# is still scanned.
DOT_DIR_IGNORE_PATTERN = re.compile(r"^\.[A-Za-z][^ ]*/$")


def _text_for_banned_scan(rel_path, text):
    if rel_path != ".gitignore":
        return text
    kept_lines = [
        line
        for line in text.splitlines()
        if not DOT_DIR_IGNORE_PATTERN.match(line.strip())
    ]
    return "\n".join(kept_lines)


BANNED_WORDS = [
    "col" + "lege",
    "course" + "work",
    "origin" + "ally",
    "course" + " project",
    "Cla" + "ude",
    "Anthro" + "pic",
    "Co-Au" + "thored-By",
    "/Us" + "ers/",
]

# The re+build/re+built family needs word-boundary matching rather than
# plain substring matching: a bare substring check flags innocuous words
# like "pre" + "built" (which shows up in vendored third-party files, e.g.
# a doc comment in @mediapipe/tasks-vision's wasm loader). Assembled from
# fragments, like BANNED_WORDS above, and phrased without the literal word
# anywhere in this comment, so this file doesn't trip its own guard.
BANNED_WORD_PATTERNS = [
    re.compile(r"\bre" + "buil" + r"[dt]\b", re.IGNORECASE),
]


def _git_ls_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return [line for line in out.stdout.splitlines() if line]


def _iter_tracked_text_files():
    for rel_path in _git_ls_files():
        if rel_path in SKIP_PATHS:
            continue
        if Path(rel_path).suffix.lower() in BINARY_EXTENSIONS:
            continue
        full_path = REPO_ROOT / rel_path
        try:
            text = full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        yield rel_path, text


def test_private_paths_are_ignored():
    private_paths = [
        "data/x",
        "private/handbook.pdf",
        "assets/x.jpg",
        "reports/parity_fixtures/f.png",
    ]
    for path in private_paths:
        result = subprocess.run(
            ["git", "check-ignore", "-q", path], cwd=REPO_ROOT
        )
        assert result.returncode == 0, f"{path} is not git-ignored"


def test_no_banned_words_in_tracked_files():
    violations = []
    for rel_path, text in _iter_tracked_text_files():
        scanned = _text_for_banned_scan(rel_path, text)
        lowered = scanned.lower()
        for banned in BANNED_WORDS:
            if banned.lower() in lowered:
                violations.append((rel_path, banned))
        for pattern in BANNED_WORD_PATTERNS:
            if pattern.search(scanned):
                violations.append((rel_path, pattern.pattern))
    assert not violations, f"banned words found in tracked files: {violations}"


def test_banned_word_pattern_positive_and_negative_examples():
    pattern = BANNED_WORD_PATTERNS[0]
    clean = "A " + "pre" + "built" + " local version of the documentation"
    flagged = "we " + "re" + "built" + " it"
    assert not pattern.search(clean)
    assert pattern.search(flagged)


def test_no_absolute_user_paths():
    needle = "/Us" + "ers/"
    violations = []
    for rel_path, text in _iter_tracked_text_files():
        if needle in text:
            violations.append(rel_path)
    assert not violations, f"absolute user paths found in tracked files: {violations}"


PROCESS_DOC_PATH_FRAGMENTS = [
    "spec/",
    "plans/",
    "handbook",
    "session-numbers",
    "subagent",
    "session-prompt",
    ".superpowers",
]

# Basename (filename without extension) checks catch process docs dropped at
# any directory level, e.g. a bare "SPEC.md" or "PLAN.md" at repo root, which
# the directory-fragment checks above (looking for "spec/", "plans/") cannot
# see. Underscores are normalized to hyphens so "PROMPT_HISTORY.md" is caught
# by the same "prompt-history" check as a hyphenated name would be.
PROCESS_DOC_BASENAME_EXACT = {"spec", "plan", "prompt"}
PROCESS_DOC_BASENAME_CONTAINS = [
    "handbook",
    "session-numbers",
    "subagent",
    "session-prompt",
    "prompt-history",
]
PROCESS_DOC_BASENAME_PREFIXES = ("spec-", "plan-")


def _is_process_doc_path(rel_path):
    lowered_path = rel_path.lower()
    for fragment in PROCESS_DOC_PATH_FRAGMENTS:
        if fragment in lowered_path:
            return True

    stem = Path(rel_path).stem.lower().replace("_", "-")
    if stem in PROCESS_DOC_BASENAME_EXACT:
        return True
    for fragment in PROCESS_DOC_BASENAME_CONTAINS:
        if fragment in stem:
            return True
    for prefix in PROCESS_DOC_BASENAME_PREFIXES:
        if stem.startswith(prefix):
            return True
    return False


def test_process_doc_matcher_positive_and_negative_examples():
    must_flag = ["SPEC.md", "PLAN.md", "docs/plans/x.md", "PROMPT_HISTORY.md"]
    must_not_flag = [
        "docs/DESIGN_CARD.md",
        "src/carmine/pigment.py",
        "tests/test_landmarks.py",
    ]
    for path in must_flag:
        assert _is_process_doc_path(path), f"matcher should flag {path}"
    for path in must_not_flag:
        assert not _is_process_doc_path(path), f"matcher should not flag {path}"


def test_no_process_docs_tracked():
    violations = [
        rel_path for rel_path in _git_ls_files() if _is_process_doc_path(rel_path)
    ]
    assert not violations, f"process/planning docs tracked: {violations}"
