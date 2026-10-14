"""Zero-token checks of the retrieval layer on a split (no LLM call).

1. Evidence hit rate for task A: share of gold entailment/contradiction cases where one of
   our first five evidence chunks overlaps the gold passage. The matching rule follows the
   description in the official starter README (normalise case, whitespace and words
   hyphenated across line breaks; rapidfuzz partial_ratio >= 90; items <= 5,000 chars).
2. Neutral gate: distribution of the best-chunk cosine for neutral vs non-neutral cases,
   and how many cases each threshold would route to "neutral" without a model call.
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lowprio  # noqa: E402,F401
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from booklet_check import pipeline  # noqa: E402
from booklet_check.llm import Apertus  # noqa: E402


def norm(t):
    t = unicodedata.normalize("NFKC", t).replace("­", "")
    t = re.sub(r"(\w) ?-\s*\n\s*(\w)", r"\1\2", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def hit(quote, passage, min_ratio=90, max_chars=5000):
    q, p = norm(quote), norm(passage)
    return bool(q) and len(q) <= max_chars and fuzz.partial_ratio(q, p) >= min_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="split dir with cases.jsonl + expected-labels.jsonl")
    ap.add_argument("--booklet-root", required=True)
    ap.add_argument("--out", help="write per-case diagnostics jsonl")
    args = ap.parse_args()
    d = Path(args.dir)
    cases = [json.loads(l) for l in (d / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = {e["id"]: e for e in map(json.loads, (d / "expected-labels.jsonl").read_text(encoding="utf-8").splitlines())}
    pred = pipeline.Predictor(Path(args.booklet_root), llm=Apertus.__new__(Apertus))
    rows = []
    for c in cases:
        if "booklet" in c and not (Path(args.booklet_root) / c["booklet"]["path"]).exists():
            continue
        try:
            passages, evidence, _, diag = pred._retrieve(c, c["claim"]["text"], c.get("vote", ""))
        except Exception as exc:  # incomplete local PDF download
            print("skip", c["id"], exc, file=sys.stderr)
            continue
        g = gold[c["id"]]
        r = {"id": c["id"], "task": c["id"][-1], "label": g["label"], "top_cos": diag.get("top_cos", 0),
             "max_cos": diag.get("max_cos", 0), "top_in_vote": diag.get("top_in_vote"),
             "ctx_chars": sum(len(p) for p in passages)}
        if r["task"] == "A" and g["label"] != 1:
            r["hit"] = any(hit(e["text"], g["reference"]) for e in evidence[:5])
            r["hit_rank"] = next((i for i, e in enumerate(evidence[:5]) if hit(e["text"], g["reference"])), None)
        rows.append(r)
    A = [r for r in rows if r["task"] == "A" and r["label"] != 1]
    if A:
        print(f"task A evidence hit: {sum(r['hit'] for r in A)}/{len(A)} = {sum(r['hit'] for r in A)/len(A):.3f}")
    for task in "AB":
        T = [r for r in rows if r["task"] == task]
        if not T:
            continue
        print(f"task {task}: {len(T)} cases, mean context chars {sum(r['ctx_chars'] for r in T)/len(T):.0f}")
        for th in (0.2, 0.25, 0.3, 0.35, 0.4):
            gated = [r for r in T if r["top_cos"] < th]
            wrong = sum(1 for r in gated if r["label"] != 1)
            neu = sum(1 for r in T if r["label"] == 1)
            print(f"  gate<{th}: routes {len(gated)} ({len(gated)/len(T):.0%}), of which non-neutral {wrong}; "
                  f"catches {len(gated)-wrong}/{neu} neutrals")
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
