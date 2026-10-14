"""Run one system on a frozen split, score it with the official evaluate.py, record everything.

    python scripts/run_eval.py --system ours --split-dir <dir> --booklet-root <dir> --tag holdout-ours-8b \
        --official-eval <path to hackapertus-starter/evaluate.py>

Writes eval/runs/<tag>/{predictions,trace}.jsonl and score.json, and appends one record to
eval/results.json, the single source for every number in eval/RESULTS.md.
Spend: every model call is logged to eval/spend.jsonl; calls stop at BUDGET_USD (default 1.50).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import _lowprio  # noqa: E402,F401

os.environ.setdefault("BUDGET_LEDGER", str(ROOT / "eval" / "spend.jsonl"))
sys.path.insert(0, str(ROOT / "src"))
from booklet_check.llm import Apertus  # noqa: E402
from booklet_check.pipeline import Predictor, safe_predict  # noqa: E402
from baseline_full_context import Baseline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["ours", "baseline"], required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--booklet-root", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--official-eval", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    os.environ["RUN_TAG"] = args.tag
    llm = Apertus()
    cases = [json.loads(l) for l in (args.split_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    cache = Path(os.environ["INDEX_CACHE"]) if os.environ.get("INDEX_CACHE") else None
    if args.system == "ours":
        system = Predictor(args.booklet_root, cache, llm)
        fn = lambda c: safe_predict(system, c)  # noqa: E731
    else:
        system = Baseline(args.booklet_root, cache, llm)
        fn = system.predict
    spent0 = llm.spent()
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(fn, cases))
    # Transient network failures (DNS, 429 storms) become neutral answers; retry those cases
    # once, sequentially, so a local outage does not end up in the scores.
    retried = 0
    for i, r in enumerate(results):
        if r.get("_trace", {}).get("route") == "error":
            time.sleep(5)
            results[i] = fn(cases[i])
            retried += 1
    wall = time.time() - t0

    out = ROOT / "eval" / "runs" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_trace"}, ensure_ascii=False) + "\n")
    with (out / "trace.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({"id": r["id"], "label": r["label"], **r.get("_trace", {}), **r["metrics"]},
                               ensure_ascii=False) + "\n")
    subprocess.run([sys.executable, str(args.official_eval), "--predictions", str(out / "predictions.jsonl"),
                    "--expected", str(args.split_dir / "expected-labels.jsonl"), "--cases",
                    str(args.split_dir / "cases.jsonl"), "--json", str(out / "score.json")], check=True)
    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    routes: dict[str, int] = {}
    for r in results:
        k = r.get("_trace", {}).get("route", "?")
        routes[k] = routes.get(k, 0) + 1
    per_task = {}
    for task in ("A", "B"):
        rs = [r for r in results if r["id"].endswith("-" + task)]
        if rs:
            per_task[task] = {k: sum(r["metrics"][k] for r in rs) / len(rs)
                              for k in ("input_tokens", "output_tokens", "inference_time_ms")}
    record = {"tag": args.tag, "system": args.system, "model": llm.model, "dry": llm.dry,
              "split": args.split_dir.name, "cases": len(cases), "wall_s": round(wall, 1),
              "usd": round(llm.spent() - spent0, 5), "routes": routes, "retried_after_error": retried,
              "truncated_cases": sum(1 for r in results if r.get("_trace", {}).get("truncated")), "mean_per_case": per_task,
              "score": score, "note": args.note, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    res_path = ROOT / "eval" / "results.json"
    allres = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else []
    allres = [r for r in allres if r["tag"] != args.tag] + [record]
    res_path.write_text(json.dumps(allres, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[{args.tag}] usd {record['usd']:.4f} total spent {llm.spent():.4f} routes {routes}")


if __name__ == "__main__":
    main()
