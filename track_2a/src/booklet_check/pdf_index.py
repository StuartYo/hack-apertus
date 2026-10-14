"""PDF text extraction and chunking. Pure Python (pypdf), no model involved.

Evidence quotes must be verbatim, so every chunk is a contiguous slice of the page text as
pypdf returns it (line breaks kept). The official scorer normalises whitespace and words
hyphenated across line breaks on both sides, so keeping the raw lines is the safest choice.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Chunk:
    page: int  # 1-based PDF page number
    idx: int  # position of the chunk inside the page
    text: str  # verbatim slice of the page text


def clean_page(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def extract_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(clean_page(page.extract_text() or ""))
        except Exception:  # a broken page must not kill the whole case
            pages.append("")
    return pages


def split_page(page_text: str, target: int = 500, hard_max: int = 900) -> list[str]:
    """Split a page into pieces of about `target` characters along line boundaries.

    A piece closes at a blank line or at a line that ends a sentence once it is long
    enough, and always before it grows past `hard_max`.
    """
    lines = page_text.split("\n")
    pieces, buf, size = [], [], 0
    for line in lines:
        if not line:
            if size >= target * 0.5:
                pieces.append("\n".join(buf))
                buf, size = [], 0
            continue
        if size + len(line) > hard_max and buf:
            pieces.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
        if size >= target and re.search(r"[.!?:»“\"]$", line):
            pieces.append("\n".join(buf))
            buf, size = [], 0
    if buf:
        pieces.append("\n".join(buf))
    return [p.strip() for p in pieces if len(p.strip()) > 30]


@dataclass
class BookletIndex:
    path: str
    pages: list[str]
    chunks: list[Chunk]

    def to_json(self) -> str:
        return json.dumps({"path": self.path, "pages": self.pages,
                           "chunks": [asdict(c) for c in self.chunks]}, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "BookletIndex":
        d = json.loads(raw)
        return cls(d["path"], d["pages"], [Chunk(**c) for c in d["chunks"]])


def file_key(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]


def chunk_pages(pages: list[str], target: int = 500, hard_max: int = 900) -> list[Chunk]:
    return [Chunk(page=i + 1, idx=j, text=t)
            for i, p in enumerate(pages) for j, t in enumerate(split_page(p, target, hard_max))]


def build_index(pdf_path: Path, cache_dir: Path | None = None) -> BookletIndex:
    """Extract and chunk a booklet. Cached by file content hash, so case order never matters."""
    key = file_key(pdf_path)
    if cache_dir:
        cached = cache_dir / f"{key}.json"
        if cached.exists():
            return BookletIndex.from_json(cached.read_text(encoding="utf-8"))
    pages = extract_pages(pdf_path)
    index = BookletIndex(str(pdf_path), pages, chunk_pages(pages))
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{key}.json").write_text(index.to_json(), encoding="utf-8")
    return index


def text_chunks(text: str, target: int = 500, hard_max: int = 900) -> list[Chunk]:
    """Chunk a plain reference text (task B). Page is None-like (0) for these chunks."""
    return [Chunk(page=0, idx=j, text=t) for j, t in enumerate(split_page(clean_page(text), target, hard_max))]


def _junk(line: str) -> bool:
    s = line.strip()
    letters = sum(ch.isalpha() for ch in s)
    return len(s) <= 3 or letters < 3 or (len(s) < 20 and not any(ch.islower() for ch in s))


def trim_edges(text: str) -> str:
    """Drop layout debris (page numbers, sidebar words, lone dashes) at the start and end of a
    chunk, so the quote stays a contiguous slice of the page but carries no foreign text."""
    lines = text.split("\n")
    while len(lines) > 1 and _junk(lines[0]):
        lines.pop(0)
    while len(lines) > 1 and _junk(lines[-1]):
        lines.pop()
    return "\n".join(lines)
