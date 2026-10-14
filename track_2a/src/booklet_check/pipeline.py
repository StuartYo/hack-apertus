"""Case -> prediction. Retrieval and evidence are code; Apertus only picks the label.

Task A (booklet PDF): extract pages, rank ~500-character chunks of the normalised pypdf text
against the claim (multilingual embeddings + BM25 + vote-section prior), show the top chunks
to Apertus and ask for one word. Evidence = the top chunks themselves, with their 1-based PDF page numbers, so a
quote can never be invented and its page is always right.
Task B (given reference): the same prompt on the reference text; long references are cut
down to the most relevant chunks first.
Neutral gate: optional. When NEUTRAL_GATE is set and the best chunk is further from the claim
than that cosine, the case is labelled neutral without a model call. It is off by default
(NEUTRAL_GATE=-1) and was off in every reported run: neutral claims in the public data come
from other votes, and a cut-off tuned on them would not carry over to same-vote neutrals.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

from . import retrieve
from .llm import Apertus, BudgetExceeded
from .pdf_index import BookletIndex, build_index, text_chunks

LABELS = ["entailment", "neutral", "contradiction"]
LANG = {"de": "German", "fr": "French", "it": "Italian", "en": "English", "rm": "Romansh"}

TOP_K = int(os.environ.get("TOP_K", "4"))  # 4 vs 8 on dev: -0.013 macro-F1 (noise at n=150), -45% tokens
EVIDENCE_ITEMS = 5
REF_MAX_CHARS = int(os.environ.get("REF_MAX_CHARS", "2600"))
NEUTRAL_GATE = float(os.environ.get("NEUTRAL_GATE", "-1"))  # -1 = gate off (default, all reported runs)

SYSTEM = ("You verify claims against the official Swiss federal voting booklet. "
          "Answer with exactly one word: entailment, contradiction, or neutral.")

INSTRUCTIONS = (
    "entailment = the text states the claim or directly implies it.\n"
    "contradiction = the text states something incompatible with the claim: another number, date, "
    "amount, actor, consequence, position or voting recommendation.\n"
    "neutral = the claim is about another subject, or the text does not settle it.\n"
    "Answer with one word."
)


def build_prompt(vote: str, passages: list[str], claim: str, claim_lang: str, src_lang: str) -> list[dict]:
    body = "\n\n".join(passages)
    user = (f"Vote: {vote}\n\nBooklet text ({LANG.get(src_lang, src_lang)}):\n{body}\n\n"
            f"Claim ({LANG.get(claim_lang, claim_lang)}): {claim}\n\n{INSTRUCTIONS}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_label(text: str) -> int | None:
    t = text.strip().lower()
    m = re.search(r"entail|contradict|neutral", t)
    if not m:
        return None
    return {"entail": 0, "neutral": 1, "contradict": 2}[m.group(0)]


class Predictor:
    def __init__(self, booklet_root: Path, cache_dir: Path | None = None, llm: Apertus | None = None):
        self.root = booklet_root
        self.cache_dir = cache_dir
        self.llm = llm or Apertus()
        self._indexes: dict[str, BookletIndex] = {}
        self._lock = threading.Lock()  # retrieval is CPU work; only model calls run in parallel

    def index(self, rel_path: str) -> BookletIndex:
        if rel_path not in self._indexes:
            self._indexes[rel_path] = build_index(self.root / rel_path, self.cache_dir)
        return self._indexes[rel_path]

    def predict(self, case: dict) -> dict:
        t0 = time.time()
        claim = case["claim"]["text"]
        claim_lang = case["claim"].get("language", "")
        vote = case.get("vote", "")
        trace: dict = {}
        with self._lock:
            passages, evidence, src_lang, diag = self._retrieve(case, claim, vote)
        trace.update(diag)
        return self._decide(case, t0, claim, claim_lang, vote, passages, evidence, src_lang, trace)

    def _retrieve(self, case: dict, claim: str, vote: str):
        if "booklet" in case:
            idx = self.index(case["booklet"]["path"])
            src_lang = case["booklet"].get("language", "")
            top, diag = retrieve.rank(idx.chunks, claim, vote, idx.pages, k=max(TOP_K, EVIDENCE_ITEMS))
            passages = [f"[p. {c.page}] {c.text}" for c, _ in top[:TOP_K]]
            evidence = [{"page": c.page, "text": c.text} for c, _ in top[:EVIDENCE_ITEMS]]
        else:
            ref = case["reference"]["text"]
            src_lang = case["reference"].get("language", "")
            chunks = text_chunks(ref)
            top, diag = retrieve.rank(chunks, claim, None, None, k=len(chunks) or 1)
            if len(ref) <= REF_MAX_CHARS or not chunks:
                passages = [ref]
            else:  # keep the best chunks up to the budget, in reading order
                keep, size = [], 0
                for c, _ in top:
                    if size + len(c.text) > REF_MAX_CHARS and keep:
                        break
                    keep.append(c)
                    size += len(c.text)
                passages = [c.text for c in sorted(keep, key=lambda c: c.idx)]
            evidence = [{"page": None, "text": c.text} for c, _ in top[:1]]
        return passages, evidence, src_lang, diag

    def _decide(self, case, t0, claim, claim_lang, vote, passages, evidence, src_lang, trace):
        diag = trace

        label, reply = None, None
        if diag.get("embedded") and diag.get("top_cos", 1.0) < NEUTRAL_GATE:
            label, trace["route"] = 1, "neutral_gate"
        else:
            reply = self.llm.chat(build_prompt(vote, passages, claim, claim_lang, src_lang))
            if reply.dry:
                label, trace["route"] = 0, "dry_placeholder"
            else:
                label = parse_label(reply.text)
                trace["route"], trace["raw"] = "model", reply.text[:40]
                if label is None:
                    label, trace["route"] = 1, "model_unparsed"
        if label == 1:
            evidence = []
        return {
            "id": case["id"],
            "label": label,
            "label_name": LABELS[label],
            "evidence": evidence,
            "metrics": {
                "input_tokens": reply.input_tokens if reply else 0,
                "output_tokens": reply.output_tokens if reply else 0,
                "inference_time_ms": int((time.time() - t0) * 1000),
            },
            "_trace": trace,
        }


def safe_predict(pred: Predictor, case: dict) -> dict:
    """Never let one case break the output file; a failure becomes a valid neutral answer."""
    try:
        return pred.predict(case)
    except BudgetExceeded:
        raise
    except Exception as exc:  # pragma: no cover
        return {"id": case.get("id"), "label": 1, "label_name": "neutral", "evidence": [],
                "metrics": {"input_tokens": 0, "output_tokens": 0, "inference_time_ms": 0},
                "_trace": {"route": "error", "error": str(exc)[:200]}}
