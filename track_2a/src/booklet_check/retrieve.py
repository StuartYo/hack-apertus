"""Hybrid retrieval over booklet chunks: multilingual embeddings + BM25 + vote-section prior.

Claims are often in another language than the booklet (de claim, fr booklet), so lexical
BM25 alone fails; it still helps with numbers, dates and names, which survive translation.
No LLM is involved here, so retrieval and the evidence quotes cost zero tokens.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter

import numpy as np

from . import embed
from .pdf_index import BookletIndex, Chunk

STOP = set("""
der die das und oder ist sind ein eine einer eines einem den dem des zu im in am an auf aus bei mit
nach von vor für fur über uber unter durch als auch nicht nur wie wird werden wurde wurden hat haben
sich es er sie wir ihr man dass daß so zum zur vom bis um kein keine soll sollen kann können muss
le la les un une des du de d l et ou est sont au aux en dans par pour sur avec sans ne pas plus que
qui quoi dont ce cet cette ces se sa son ses leur leurs il elle ils elles on a été être avoir doit
il lo la i gli le un uno una dei degli delle del dello della di da e o è sono al allo alla ai agli
alle in nel nello nella nei negli nelle con su per tra fra non più che chi cui questo questa questi
queste si suo sua suoi sue loro ha hanno essere stato stata deve
""".split())


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower().replace("­", "")
    # unify thousands separators: 1'000, 1’000, 1 000 -> 1000
    text = re.sub(r"(?<=\d)['’ ](?=\d{3}\b)", "", text)
    out = []
    for tok in re.findall(r"\d+(?:[.,]\d+)?|[^\W\d_]+", text):
        if tok in STOP:
            continue
        if tok[0].isdigit():
            out.append(tok.replace(",", "."))
            continue
        tok = _strip_accents(tok)
        if len(tok) < 3:
            continue
        out.append(tok[:7])  # crude prefix stemming works for de/fr/it compounds and inflections
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.avgdl = sum(len(d) for d in docs) / max(1, len(docs))
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        q = Counter(query)
        res = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * len(d) / (self.avgdl or 1))
            for t, qn in q.items():
                f = tf.get(t)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + norm) * (1 + 0.2 * (qn - 1))
            res.append(s)
        return res


def vote_pages(index: BookletIndex, vote: str, window: int = 2) -> set[int]:
    """Pages that belong to the section about `vote` (title keyword density, plus neighbours)."""
    vt = [t for t in tokenize(vote) if not t[0].isdigit()]
    if not vt:
        return set(range(1, len(index.pages) + 1))
    page_tokens = [Counter(tokenize(p)) for p in index.pages]
    vset = set(vt)
    dens = [sum(1 for t in vset if pt.get(t)) / len(vset) for pt in page_tokens]
    best = max(dens) if dens else 0
    keep = {i + 1 for i, d in enumerate(dens) if d >= max(0.6, best * 0.8)}
    grown = set(keep)
    for p in keep:
        grown.update(range(max(1, p - window), min(len(index.pages), p + window) + 1))
    return grown


# --- booklet structure -------------------------------------------------------------------
# Claims name their source ("the committee says", "according to the summary", "the voting
# text states"), and every booklet has the same sections in de/fr/it. Tagging pages with
# their section and boosting the matching section is a zero-token structural prior.
CLAIM_CUES = [
    ("context", r"zusammenfassung|r[ée]sum[ée]|riassunto|riepilogo|si le vote|si la votation|wird die abstimmung|"
                r"bei annahme|se il voto|se la votazione"),
    ("votetext", r"abstimmungstext|\btexte\b|\btesto\b|\btesti\b"),
    ("committee", r"komitee|comit[ée]|comitato|comitati"),
    ("council", r"bundesrat|conseil f[ée]d[ée]ral|consiglio federale"),
]
_ARG = r"ar(?:gu|go)ment\w*\W+(?:\w+\W+){0,3}?"
# Markers are searched in the first lines of a page (after the running header).
PAGE_MARKERS = [
    ("committee", r"stammt vom|stammen von den|auteurs? du texte|autor[ei] del testo|"
                  r"^(?:das |le |il |i )?(?:initiativ|referendums)?(?:komitee|comit[ée]s?|comitat[oi])\b|"
                  + _ARG + r"(?:initiativ|referendums)?(?:komitee|comit)"),
    ("council", r"^(?:bundesrat und parlament|conseil f[ée]d[ée]ral et parlement|consiglio federale e parlamento)|"
                + _ARG + r"(?:bundesrat|conseil f[ée]d[ée]ral|consiglio federale)"),
    ("votetext", r"^(?:der |le |les |il |i )?(?:abstimmungstext|textes? soumis au vote|test[oi] in votazione|"
                 r"test[oi] sottopost)"),
    ("context", r"ausgangslage|contexte|situazione iniziale|contesto|in k[üu]rze|en bref|in breve"),
]
_TOC = r"im detail|en d[ée]tail|in dettaglio"
_RUNNING = r"vorlage:|objet ?:|oggetto:"


def _head_lines(page: str, n: int = 5) -> list[str]:
    lines = [l.strip().lower() for l in page.split("\n") if len(l.strip()) > 2]
    lines = [l for l in lines if not re.search(_RUNNING, l)]
    return [re.sub(r"^\d+\s*", "", l) for l in lines[:n]]


def claim_cue(claim: str) -> str | None:
    c = claim.lower()
    for name, pat in CLAIM_CUES:
        if re.search(pat, c):
            return name
    return None


def page_sections(pages: list[str]) -> list[set[str]]:
    """Section tags per page from markers in the page head; committee / council / voting-text
    tags also cover the next page (those sections are laid out as double pages). A page head
    with several different markers is a table of contents and gets no tag."""
    tags: list[set[str]] = []
    for p in pages:
        lines = _head_lines(p)
        joined = " | ".join(lines)
        found = {name for name, pat in PAGE_MARKERS
                 if any(re.search(pat, l) for l in lines)}
        if re.search(_TOC, joined) or ("committee" in found and "council" in found):
            found = set()  # overview / table of contents
        elif len(found & {"committee", "council", "votetext"}) > 1:
            found = set()
        if found & {"committee", "council"}:
            found.discard("context")
        tags.append(found)
    out = [set(t) for t in tags]
    for i in range(1, len(tags)):
        if not tags[i] & {"committee", "council", "votetext"}:
            out[i] |= tags[i - 1] & {"committee", "council", "votetext"}
    return out


_SECTION_CACHE: dict[int, list[set[str]]] = {}


def sections_for(pages: list[str]) -> list[set[str]]:
    k = hash(tuple(pages))
    if k not in _SECTION_CACHE:
        _SECTION_CACHE[k] = page_sections(pages)
    return _SECTION_CACHE[k]


_EMB_CACHE: dict[int, np.ndarray | None] = {}


_BM25_CACHE: dict[int, BM25] = {}
_VOTE_CACHE: dict[tuple, set[int]] = {}


def _key(chunks: list[Chunk]) -> int:
    return hash(tuple(c.text for c in chunks))


def chunk_vectors(chunks: list[Chunk], key: int | None = None) -> np.ndarray | None:
    key = key if key is not None else _key(chunks)
    if key not in _EMB_CACHE:
        _EMB_CACHE[key] = embed.encode([c.text for c in chunks])
    return _EMB_CACHE[key]


def chunk_bm25(chunks: list[Chunk], key: int) -> BM25:
    if key not in _BM25_CACHE:
        _BM25_CACHE[key] = BM25([tokenize(c.text) for c in chunks])
    return _BM25_CACHE[key]


def vote_pages_from_pages(pages: list[str], vote: str, window: int = 2) -> set[int]:
    k = (hash(tuple(pages)), vote)
    if k not in _VOTE_CACHE:
        _VOTE_CACHE[k] = vote_pages(BookletIndex("", pages, []), vote, window)
    return _VOTE_CACHE[k]


SECTION_BOOST = float(os.environ.get("SECTION_BOOST", "0.3"))


def rank(chunks: list[Chunk], claim: str, vote: str | None = None, pages: list[str] | None = None,
         k: int = 6, w_bm25: float = 0.35, vote_boost: float = 0.08,
         section_boost: float | None = None) -> tuple[list[tuple[Chunk, float]], dict]:
    """Rank chunks for a claim. Returns the top-k and diagnostics used by the neutral gate."""
    if not chunks:
        return [], {"max_cos": 0.0, "top_in_vote": False, "embedded": False}
    key = _key(chunks)
    bm = chunk_bm25(chunks, key).scores(tokenize(claim))
    bmax = max(bm) or 1.0
    vecs = chunk_vectors(chunks, key)
    if vecs is not None:
        q = embed.encode([claim], kind="query")[0]
        cos = (vecs @ q).tolist()
    else:
        cos = [0.0] * len(chunks)
    in_vote = [False] * len(chunks)
    if vote and pages:
        vp = vote_pages_from_pages(pages, vote)
        in_vote = [c.page in vp for c in chunks]
    total = [cs + w_bm25 * b / bmax + (vote_boost if iv else 0.0) for cs, b, iv in zip(cos, bm, in_vote)]
    cue = claim_cue(claim)
    sb = SECTION_BOOST if section_boost is None else section_boost
    if cue and pages and sb:
        secs = sections_for(pages)
        total = [t + (sb if cue in secs[c.page - 1] else 0.0) for t, c in zip(total, chunks)]
    order = sorted(range(len(chunks)), key=lambda i: -total[i])
    top = [(chunks[i], total[i]) for i in order[:k]]
    diag = {"cue": cue, "max_cos": max(cos), "top_cos": cos[order[0]], "top_in_vote": in_vote[order[0]],
            "embedded": vecs is not None}
    return top, diag
