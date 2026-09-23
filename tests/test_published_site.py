"""The published page carries its authorship marks.

GitHub Pages serves `docs/`, which Vite builds from `web/index.html` and
`web/src/`. Every mark is checked twice: in the source that generates the page,
because a mark that exists only in `docs/` disappears on the next build, and in
the committed build, because a mark that exists only in the source says nothing
about what is actually being served.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "web" / "index.html"
BUILT = REPO_ROOT / "docs" / "index.html"
ENTRY = REPO_ROOT / "web" / "src" / "main.ts"
SECTIONS = REPO_ROOT / "web" / "src" / "ui" / "sections.ts"

AUTHOR = "Safdar Hussain"
SOURCE_COMMENT = "<!-- Carmine · built by Safdar Hussain · https://github.com/safdar-hussain1 -->"
CONSOLE_LINE = (
    'console.info("Carmine — built by Safdar Hussain · https://github.com/safdar-hussain1/carmine")'
)
FOOTER_CREDIT = (
    '<a href="https://github.com/safdar-hussain1" rel="author">Built by Safdar Hussain</a>'
)


class _Page(HTMLParser):
    """Collects the `<meta>` tags and the module script of one HTML page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.module_scripts: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "meta":
            key = attributes.get("name") or attributes.get("property")
            if key:
                self.meta.setdefault(key, attributes.get("content") or "")
        elif tag == "script" and attributes.get("type") == "module" and attributes.get("src"):
            self.module_scripts.append(attributes["src"])


def _parse(path: Path) -> tuple[str, _Page]:
    text = path.read_text(encoding="utf-8")
    page = _Page()
    page.feed(text)
    return text, page


def _function_body(source: str, name: str) -> str:
    """The body of a top-level TypeScript function, up to its closing brace."""
    match = re.search(rf"^(?:export )?function {name}\(\)[^{{]*\{{\n(.*?)^\}}", source, re.M | re.S)
    assert match, f"function {name}() not found"
    return match.group(1)


def _built_bundle() -> str:
    """The JavaScript bundle docs/index.html actually loads."""
    _, page = _parse(BUILT)
    assert len(page.module_scripts) == 1, page.module_scripts
    bundle = (BUILT.parent / page.module_scripts[0]).resolve()
    assert bundle.is_relative_to(BUILT.parent.resolve()), bundle
    return bundle.read_text(encoding="utf-8")


def test_template_carries_the_author_meta_and_source_comment():
    text, page = _parse(TEMPLATE)
    assert page.meta.get("author") == AUTHOR
    assert SOURCE_COMMENT in text


def test_app_entry_signs_the_console():
    assert CONSOLE_LINE in _function_body(ENTRY.read_text(encoding="utf-8"), "bootstrap")


def test_footer_credits_the_author():
    assert FOOTER_CREDIT in _function_body(SECTIONS.read_text(encoding="utf-8"), "footerHtml")


def test_built_page_carries_every_mark():
    text, page = _parse(BUILT)
    assert page.meta.get("author") == AUTHOR
    assert SOURCE_COMMENT in text
    bundle = _built_bundle()
    assert CONSOLE_LINE in bundle
    assert FOOTER_CREDIT in bundle
