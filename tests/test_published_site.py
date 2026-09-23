"""The published page carries its authorship marks and its search metadata.

GitHub Pages serves `docs/`, which Vite builds from `web/index.html`,
`web/public/` and `web/src/`. Everything here is checked twice: in the source
that generates the page, because anything that exists only in `docs/`
disappears on the next build, and in the committed build, because anything
that exists only in the source says nothing about what is actually served.
"""

from __future__ import annotations

import json
import re
import struct
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "web" / "index.html"
BUILT = REPO_ROOT / "docs" / "index.html"
ENTRY = REPO_ROOT / "web" / "src" / "main.ts"
SECTIONS = REPO_ROOT / "web" / "src" / "ui" / "sections.ts"
PUBLIC = REPO_ROOT / "web" / "public"
WEB_SOURCES = REPO_ROOT / "web" / "src"

SITE_URL = "https://safdar-hussain1.github.io/carmine/"
SEARCH_CONSOLE_TOKEN = "0SIEfExLTSQj1qvnHWF5A5fY58KVl2lpIEnePP9CtI0"
OG_IMAGE_URL = SITE_URL + "og-image.png"
PERSON = {
    "@type": "Person",
    "name": "Safdar Hussain",
    "url": "https://github.com/safdar-hussain1",
    "sameAs": [
        "https://github.com/safdar-hussain1",
        "https://www.linkedin.com/in/safdar-hussain-a8a61b248",
    ],
}
PAGES = pytest.mark.parametrize("path", [TEMPLATE, BUILT], ids=["template", "built"])

AUTHOR = "Safdar Hussain"
SOURCE_COMMENT = "<!-- Carmine · built by Safdar Hussain · https://github.com/safdar-hussain1 -->"
CONSOLE_LINE = (
    'console.info("Carmine — built by Safdar Hussain · https://github.com/safdar-hussain1/carmine")'
)
FOOTER_CREDIT = (
    '<a href="https://github.com/safdar-hussain1" rel="author">Built by Safdar Hussain</a>'
)


class _Page(HTMLParser):
    """Collects what search engines and link previews read from one HTML page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lang: str | None = None
        self.titles: list[str] = []
        self.meta: dict[str, list[str]] = {}
        self.links: list[dict[str, str]] = []
        self.json_ld: list[str] = []
        self.module_scripts: list[str] = []
        self.h1: list[str] = []
        self._capture: str | None = None
        self._buffer = ""
        self._in_head = False

    def handle_starttag(self, tag, attrs):
        attributes = {key: value or "" for key, value in attrs}
        if tag == "html":
            self.lang = attributes.get("lang")
        elif tag == "head":
            self._in_head = True
        elif tag == "meta":
            key = attributes.get("name") or attributes.get("property")
            if key:
                self.meta.setdefault(key, []).append(attributes.get("content", ""))
        elif tag == "link":
            self.links.append(attributes)
        elif tag == "script" and attributes.get("type") == "module" and attributes.get("src"):
            self.module_scripts.append(attributes["src"])
        elif tag == "script" and attributes.get("type") == "application/ld+json":
            self._capture, self._buffer = "ld", ""
        elif tag == "title" and self._in_head:
            self._capture, self._buffer = "title", ""
        elif tag == "h1":
            self._capture, self._buffer = "h1", ""

    def handle_endtag(self, tag):
        if tag == "head":
            self._in_head = False
        elif self._capture == "ld" and tag == "script":
            self.json_ld.append(self._buffer)
            self._capture = None
        elif self._capture == "title" and tag == "title":
            self.titles.append(" ".join(self._buffer.split()))
            self._capture = None
        elif self._capture == "h1" and tag == "h1":
            self.h1.append(" ".join(self._buffer.split()))
            self._capture = None

    def handle_data(self, data):
        if self._capture:
            self._buffer += data

    def one(self, key: str) -> str:
        values = self.meta.get(key, [])
        assert len(values) == 1, f"expected exactly one {key!r} meta tag, found {values}"
        return values[0]


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
    assert page.one("author") == AUTHOR
    assert SOURCE_COMMENT in text


def test_app_entry_signs_the_console():
    assert CONSOLE_LINE in _function_body(ENTRY.read_text(encoding="utf-8"), "bootstrap")


def test_footer_credits_the_author():
    assert FOOTER_CREDIT in _function_body(SECTIONS.read_text(encoding="utf-8"), "footerHtml")


def test_built_page_carries_every_mark():
    text, page = _parse(BUILT)
    assert page.one("author") == AUTHOR
    assert SOURCE_COMMENT in text
    bundle = _built_bundle()
    assert CONSOLE_LINE in bundle
    assert FOOTER_CREDIT in bundle


# --- search and link-preview metadata ---------------------------------------


@PAGES
def test_head_carries_title_description_and_search_console_tag(path):
    _, page = _parse(path)
    assert page.lang == "en"
    assert len(page.titles) == 1, page.titles
    title = page.titles[0]
    assert title.startswith("Carmine — ") and len(title) <= 60, title
    description = page.one("description")
    assert 120 <= len(description) <= 160, len(description)
    assert page.one("author") == AUTHOR
    assert page.one("google-site-verification") == SEARCH_CONSOLE_TOKEN
    assert "viewport" in page.meta


@PAGES
def test_canonical_and_favicon_resolve(path):
    _, page = _parse(path)
    # Relative links resolve next to the built page; the template's public
    # files live in web/public/ until the build copies them across.
    root = PUBLIC if path == TEMPLATE else BUILT.parent
    canonical = [link["href"] for link in page.links if link.get("rel") == "canonical"]
    assert canonical == [SITE_URL]
    icons = [link["href"] for link in page.links if "icon" in link.get("rel", "").split()]
    assert icons == ["favicon.svg"]
    assert (root / icons[0]).read_text(encoding="utf-8").startswith("<svg ")


@PAGES
def test_open_graph_and_twitter_card_agree_with_the_head(path):
    _, page = _parse(path)
    title, description = page.titles[0], page.one("description")
    assert page.one("og:type") == "website"
    assert page.one("og:site_name") == AUTHOR
    assert page.one("og:url") == SITE_URL
    assert page.one("og:title") == page.one("twitter:title") == title
    assert page.one("og:description") == page.one("twitter:description") == description
    assert page.one("og:image") == page.one("twitter:image") == OG_IMAGE_URL
    assert (page.one("og:image:width"), page.one("og:image:height")) == ("1200", "630")
    assert page.one("og:image:alt") == page.one("twitter:image:alt")
    assert len(page.one("og:image:alt")) > 40
    assert page.one("twitter:card") == "summary_large_image"


@PAGES
def test_structured_data_names_the_app_the_code_and_the_author(path):
    _, page = _parse(path)
    assert len(page.json_ld) == 1, "expected exactly one JSON-LD block"
    graph = json.loads(page.json_ld[0])["@graph"]
    nodes = {node["@type"]: node for node in graph}
    assert set(nodes) == {"WebApplication", "SoftwareSourceCode"}

    app = nodes["WebApplication"]
    assert app["name"] == "Carmine"
    assert app["url"] == SITE_URL
    assert app["description"] == page.one("description")
    assert app["image"] == OG_IMAGE_URL
    assert app["applicationCategory"] == "MultimediaApplication"
    for requirement in ("JavaScript", "WebGL2", "camera"):
        assert requirement in app["browserRequirements"], requirement
    assert app["isAccessibleForFree"] is True

    code = nodes["SoftwareSourceCode"]
    assert code["codeRepository"] == "https://github.com/safdar-hussain1/carmine"
    assert code["license"] == "https://opensource.org/licenses/MIT"
    for node in graph:
        assert node["author"] == PERSON, node["@type"]


def _source_h1_text() -> str:
    """The text of the one h1 the page's sources render."""
    found = []
    for path in sorted(WEB_SOURCES.rglob("*.ts")):
        if path.name.endswith(".test.ts"):
            continue
        for match in re.finditer(r"<h1[\s>](.*?)</h1>", path.read_text(encoding="utf-8"), re.S):
            found.append((path.relative_to(REPO_ROOT).as_posix(), match.group(1)))
    assert len(found) == 1, f"the page's sources should render exactly one h1, found {found}"
    return " ".join(re.sub(r"<[^>]+>", "", found[0][1]).split())


def test_built_page_serves_exactly_the_one_h1_the_app_renders():
    _, page = _parse(BUILT)
    assert page.h1 == [_source_h1_text()]


def test_template_leaves_the_app_root_empty_for_the_build_to_fill():
    _, page = _parse(TEMPLATE)
    assert page.h1 == []
    assert '<div id="app"></div>' in TEMPLATE.read_text(encoding="utf-8")


def _png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return struct.unpack(">II", data[16:24])


@pytest.mark.parametrize("name", ["og-image.png", "favicon.svg", "sitemap.xml"])
def test_public_files_are_published_unchanged(name):
    source = PUBLIC / name
    assert source.is_file(), f"web/public/{name} is missing"
    assert (BUILT.parent / name).read_bytes() == source.read_bytes()


def test_og_image_is_a_1200_by_630_png_under_500_kb():
    data = (PUBLIC / "og-image.png").read_bytes()
    assert _png_size(data) == (1200, 630)
    assert len(data) <= 500 * 1024, f"{len(data) // 1024} KB"


def test_sitemap_lists_the_page_with_a_literal_date():
    root = ET.fromstring((PUBLIC / "sitemap.xml").read_bytes())
    namespace = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = root.findall("s:url", namespace)
    assert len(urls) == 1
    assert urls[0].findtext("s:loc", namespaces=namespace) == SITE_URL
    lastmod = urls[0].findtext("s:lastmod", namespaces=namespace) or ""
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", lastmod), lastmod


def test_no_per_project_robots_txt():
    """Crawlers only read robots.txt at the host root, never under /carmine/."""
    assert not (PUBLIC / "robots.txt").exists()
    assert not (BUILT.parent / "robots.txt").exists()
