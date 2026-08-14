"""Structural guard tests.

These tests protect repository hygiene invariants: private/dataset paths stay
git-ignored, no leaked local file paths, no process/planning documents get
tracked, and no references to AI tooling end up in tracked text files.

Banned strings are assembled from concatenated fragments so this file never
contains a literal match of what it is guarding against.
"""

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
    # .gitignore must itself contain ignore patterns for AI-tool workspace
    # directories (e.g. a dotfolder named after this vendor); that is
    # infrastructure, not prose mentioning the vendor, so it is exempt here.
    ".gitignore",
}

BANNED_WORDS = [
    "re" + "build",
    "re" + "built",
    "col" + "lege",
    "course" + "work",
    "origin" + "ally",
    "course" + " project",
    "Cla" + "ude",
    "Anthro" + "pic",
    "Co-Au" + "thored-By",
    "/Us" + "ers/",
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
        lowered = text.lower()
        for banned in BANNED_WORDS:
            if banned.lower() in lowered:
                violations.append((rel_path, banned))
    assert not violations, f"banned words found in tracked files: {violations}"


def test_no_absolute_user_paths():
    needle = "/Us" + "ers/"
    violations = []
    for rel_path, text in _iter_tracked_text_files():
        if needle in text:
            violations.append(rel_path)
    assert not violations, f"absolute user paths found in tracked files: {violations}"


def test_no_process_docs_tracked():
    banned_path_fragments = [
        "spec/",
        "plans/",
        "handbook",
        "session-numbers",
        "subagent",
        "session-prompt",
        ".superpowers",
    ]
    violations = []
    for rel_path in _git_ls_files():
        lowered = rel_path.lower()
        for fragment in banned_path_fragments:
            if fragment in lowered:
                violations.append((rel_path, fragment))
    assert not violations, f"process/planning docs tracked: {violations}"
