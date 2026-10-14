"""Cut the official prepare_cases.py output into the frozen dev / held-out split.

Inputs are the files written by the starter's prepare_cases.py (all rows, both tasks,
booklets downloaded). Row numbers come from data/split.json, which was frozen before any
model was run. --sample N draws a label-stratified sample per task with a fixed seed, for
runs that would otherwise cost too many tokens (the full-booklet baseline).
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def row_of(case_id):  # v1.1-row-123-A
    return int(case_id.split("-row-")[1].split("-")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--expected", required=True)
    ap.add_argument("--split", default="data/split.json")
    ap.add_argument("--part", choices=["dev", "holdout"], required=True)
    ap.add_argument("--task", choices=["A", "B", "both"], default="both")
    ap.add_argument("--sample", type=int, help="cases per task, label-stratified")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rows = set(json.loads(Path(args.split).read_text(encoding="utf-8"))["rows"][args.part])
    gold = {e["id"]: e for e in read(args.expected)}
    cases = [c for c in read(args.cases) if row_of(c["id"]) in rows
             and (args.task == "both" or c["id"].endswith("-" + args.task))]
    if args.sample:
        rng = random.Random(args.seed)
        picked = []
        for task in ("A", "B"):
            by_label = defaultdict(list)
            for c in cases:
                if c["id"].endswith("-" + task):
                    by_label[gold[c["id"]]["label"]].append(c)
            if not by_label:
                continue
            per = args.sample // len(by_label)
            for lab in sorted(by_label):
                pool = sorted(by_label[lab], key=lambda c: c["id"])
                picked += rng.sample(pool, min(per, len(pool)))
        cases = sorted(picked, key=lambda c: (row_of(c["id"]), c["id"]))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    (out / "expected-labels.jsonl").write_text(
        "".join(json.dumps(gold[c["id"]], ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    print(f"{args.part}: {len(cases)} cases -> {out}")


if __name__ == "__main__":
    main()
