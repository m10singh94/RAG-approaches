"""Download the public corpus and normalise it to plain text.

Both the DIY index and the Knowledge Base ingest the *same* processed .txt files, so retrieval quality is
compared on identical input. (Knowledge Bases can parse PDFs itself; that's a separate experiment.)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import requests
import yaml
from bs4 import BeautifulSoup
from pypdf import PdfReader

from rag_approaches.config import DATA_DIR, ROOT

MANIFEST = ROOT / "corpus" / "manifest.yaml"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
HEADERS = {"User-Agent": "rag-approaches-demo/0.1 (+https://github.com/m10singh94/RAG-approaches)"}
SKIP_SLUGS = {"table-of-contents", "bibliography", "index", "editors", "colophon"}
BLOCK_TAGS = ["h1", "h2", "h3", "h4", "p", "li", "pre", "blockquote", "tr", "dt", "dd", "figcaption"]


@dataclass
class Document:
    doc_id: str
    source: str
    title: str
    path: Path

    @property
    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")


def load_manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["sources"]


def _get(url: str) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    return resp


def download(force: bool = False) -> None:
    for source, spec in load_manifest().items():
        for doc in spec["documents"]:
            if doc["type"] == "pdf":
                target = RAW_DIR / source / f"{doc['id']}.pdf"
                if target.exists() and not force:
                    print(f"  skip {target.relative_to(ROOT)} (exists)")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                print(f"  GET {doc['url']}")
                target.write_bytes(_get(doc["url"]).content)
            elif doc["type"] == "html_toc":
                _download_html_toc(source, doc, force)


def _download_html_toc(source: str, doc: dict, force: bool) -> None:
    out_dir = RAW_DIR / source / doc["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    toc = BeautifulSoup(_get(doc["url"]).text, "html.parser")
    links = []
    for a in toc.find_all("a", href=True):
        href = a["href"]
        slug = href.removeprefix(doc["link_prefix"]).strip("/")
        is_chapter = slug and "/" not in slug and slug not in SKIP_SLUGS and not slug.startswith("part-")
        if href.startswith(doc["link_prefix"]) and is_chapter and slug not in links:
            links.append(slug)
    for slug in links:
        target = out_dir / f"{slug}.html"
        if target.exists() and not force:
            continue
        url = urljoin(doc["url"], f"{doc['link_prefix']}{slug}/")
        print(f"  GET {url}")
        target.write_bytes(_get(url).content)  # raw bytes: sre.google sends no charset, so .text would mis-decode
        time.sleep(0.5)  # be polite to sre.google


def _clean(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _write(source: str, doc_id: str, title: str, text: str) -> None:
    out = PROCESSED_DIR / source / f"{doc_id}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_clean(text), encoding="utf-8")
    # Sidecar metadata in the format Bedrock Knowledge Bases expects; the DIY indexer reads it too.
    meta = {"metadataAttributes": {"source": source, "title": title, "doc_id": doc_id}}
    out.with_name(out.name + ".metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def prepare() -> None:
    for source, spec in load_manifest().items():
        for doc in spec["documents"]:
            if doc["type"] == "pdf":
                pdf = RAW_DIR / source / f"{doc['id']}.pdf"
                print(f"  extract {pdf.name}")
                reader = PdfReader(pdf)
                _write(source, doc["id"], doc["title"], "\n\n".join(p.extract_text() or "" for p in reader.pages))
            elif doc["type"] == "html_toc":
                for html in sorted((RAW_DIR / source / doc["id"]).glob("*.html")):
                    if html.stem in SKIP_SLUGS:
                        continue
                    soup = BeautifulSoup(html.read_bytes(), "html.parser", from_encoding="utf-8")
                    content = soup.find(id="content") or soup.find("main") or soup.body
                    for tag in content.find_all(["script", "style", "nav"]):
                        tag.decompose()
                    for tag in content.find_all(BLOCK_TAGS):
                        tag.append("\n\n")  # paragraph breaks only at block ends, so inline links don't split lines
                    h1 = content.find(["h1", "h2"])
                    chapter = h1.get_text(" ", strip=True) if h1 else html.stem
                    _write(source, f"{doc['id']}-{html.stem}", f"{doc['title']}: {chapter}", content.get_text(" "))
    print(f"Processed {len(list(iter_documents()))} documents into {PROCESSED_DIR.relative_to(ROOT)}")


def iter_documents():
    for meta_file in sorted(PROCESSED_DIR.rglob("*.txt.metadata.json")):
        attrs = json.loads(meta_file.read_text(encoding="utf-8"))["metadataAttributes"]
        yield Document(attrs["doc_id"], attrs["source"], attrs["title"], meta_file.with_name(meta_file.name[:-14]))
