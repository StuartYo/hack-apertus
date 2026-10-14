"""CLI required by the OST challenge: --input cases.jsonl --output predictions.jsonl."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .pipeline import Predictor, safe_predict


def main(argv=None):
    ap = argparse.ArgumentParser(prog="booklet-check")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--booklet-root", type=Path, help="directory booklet.path is relative to (default: input dir)")
    ap.add_argument("--trace", type=Path, help="also write routing diagnostics per case (local analysis)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "4")))
    args = ap.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        ap.error("Input and output must be different files.")

    cases = [json.loads(l) for l in args.input.read_text(encoding="utf-8").splitlines() if l.strip()]
    root = args.booklet_root or Path(os.environ.get("BOOKLET_ROOT", args.input.parent))
    cache = Path(os.environ["INDEX_CACHE"]) if os.environ.get("INDEX_CACHE") else None
    pred = Predictor(root, cache)
    print(f"[booklet-check] {len(cases)} cases, model {pred.llm.model}{' (DRY MODE)' if pred.llm.dry else ''}",
          file=sys.stderr)
    with ThreadPoolExecutor(max(1, args.workers)) as ex:
        results = list(ex.map(lambda c: safe_predict(pred, c), cases))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps({k: v for k, v in r.items() if k != "_trace"}, ensure_ascii=False) + "\n")
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        with args.trace.open("w", encoding="utf-8") as out:
            for r in results:
                out.write(json.dumps({"id": r["id"], "label": r["label"], **r.get("_trace", {}),
                                      **r["metrics"]}, ensure_ascii=False) + "\n")
    routes = {}
    for r in results:
        routes[r.get("_trace", {}).get("route", "?")] = routes.get(r.get("_trace", {}).get("route", "?"), 0) + 1
    print(f"[booklet-check] wrote {len(results)} predictions to {args.output}; routes {routes}", file=sys.stderr)


if __name__ == "__main__":
    main()
