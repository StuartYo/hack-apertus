"""Pure-Apertus baseline: the whole booklet in the prompt, no retrieval, model quotes its own evidence.

Same model, same endpoint and the same label definitions as our system; the only difference
is that the model reads the full booklet text (page markers included) and must pick the page
and the quote itself. Booklets longer than MAX_CHARS are cut at the end (reported as
`truncated` in the trace). Task B gets the full reference with the same minimal prompt.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from booklet_check.llm import Apertus  # noqa: E402
from booklet_check.pdf_index import build_index  # noqa: E402
from booklet_check.pipeline import INSTRUCTIONS, LABELS, LANG, SYSTEM, parse_label  # noqa: E402

MAX_CHARS = int(os.environ.get("BASELINE_MAX_CHARS", "180000"))
ANSWER = ('Reply with JSON only: {"label": "entailment|contradiction|neutral", "page": <page number or null>, '
          '"quote": "<verbatim sentence from the text that decides it, or empty>"}')


class Baseline:
    def __init__(self, booklet_root: Path, cache_dir: Path | None = None, llm: Apertus | None = None):
        self.root, self.cache_dir = booklet_root, cache_dir
        self.llm = llm or Apertus()
        self._docs: dict[str, tuple[str, bool]] = {}

    def _doc(self, rel: str) -> tuple[str, bool]:
        if rel not in self._docs:
            idx = build_index(self.root / rel, self.cache_dir)
            full = "\n\n".join(f"=== page {i + 1} ===\n{p}" for i, p in enumerate(idx.pages))
            self._docs[rel] = (full[:MAX_CHARS], len(full) > MAX_CHARS)
        return self._docs[rel]

    def predict(self, case: dict) -> dict:
        t0 = time.time()
        claim, cl = case["claim"]["text"], case["claim"].get("language", "")
        if "booklet" in case:
            text, truncated = self._doc(case["booklet"]["path"])
            src = case["booklet"].get("language", "")
        else:
            text, truncated, src = case["reference"]["text"], False, case["reference"].get("language", "")
        user = (f"Vote: {case.get('vote', '')}\n\nBooklet text ({LANG.get(src, src)}):\n{text}\n\n"
                f"Claim ({LANG.get(cl, cl)}): {claim}\n\n{INSTRUCTIONS}\n{ANSWER}")
        try:
            reply = self.llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                  max_tokens=160)
        except RuntimeError as exc:  # e.g. context length exceeded: counts as a failed case
            if "spend cap" in str(exc):
                raise
            return {"id": case["id"], "label": 1, "label_name": "neutral", "evidence": [],
                    "metrics": {"input_tokens": 0, "output_tokens": 0,
                                "inference_time_ms": int((time.time() - t0) * 1000)},
                    "_trace": {"route": "error", "error": str(exc)[:200], "truncated": truncated}}
        label, page, quote = None, None, ""
        m = re.search(r"\{.*\}", reply.text, re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                label = parse_label(str(d.get("label", "")))
                page = d.get("page") if isinstance(d.get("page"), int) else None
                quote = str(d.get("quote") or "")
            except Exception:
                pass
        if label is None:
            label = parse_label(reply.text)
        route = "model" if label is not None else "model_unparsed"
        if reply.dry:
            label, route = 0, "dry_placeholder"
        label = 1 if label is None else label
        evidence = []
        if label != 1 and quote:
            evidence = [{"page": page if "booklet" in case else None, "text": quote[:5000]}]
        return {"id": case["id"], "label": label, "label_name": LABELS[label], "evidence": evidence,
                "metrics": {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                            "inference_time_ms": int((time.time() - t0) * 1000)},
                "_trace": {"route": route, "truncated": truncated, "raw": reply.text[:200]}}
