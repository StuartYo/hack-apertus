"""Derived numbers that need the gold labels: eval/analysis.json.

    python scripts/make_analysis.py --work <dir> --official-eval <hackapertus-starter/evaluate.py>

<dir> holds the split folders written by split_cases.py (holdout/, holdout60/, devA150b/), each
with cases.jsonl and expected-labels.jsonl. The gold labels are not stored in this repository,
so this script is run once locally; scripts/make_results.py then reads eval/analysis.json
and fills the numbers into the technical report, and `--check` fails if they drift.
No model is called here: everything is recomputed from the committed prediction files.
"""

import argparse
import json
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lowprio  # noqa: E402,F401

RUNS = ROOT / "eval" / "runs"
RESAMPLES = 10000
SEED = 0


def jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def by_id(rows):
    return {r["id"]: r for r in rows}


def macro_f1(ids, pred, gold):
    f = []
    for c in range(3):
        tp = sum(pred[i] == c and gold[i] == c for i in ids)
        fp = sum(pred[i] == c and gold[i] != c for i in ids)
        fn = sum(pred[i] != c and gold[i] == c for i in ids)
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f) / 3


def boot(ids, stat):
    """95% percentile interval of stat over case resamples (paired: same resample for both systems)."""
    rng = random.Random(SEED)
    vals = sorted(stat([rng.choice(ids) for _ in ids]) for _ in range(RESAMPLES))
    return [round(vals[int(0.025 * RESAMPLES)], 4), round(vals[int(0.975 * RESAMPLES) - 1], 4)]


def labels(tag):
    return {r["id"]: r["label"] for r in jsonl(RUNS / tag / "predictions.jsonl")}


def holdout60(work):
    gold = {r["id"]: r["label"] for r in jsonl(work / "holdout60" / "expected-labels.jsonl")}
    base, ours = labels("holdout60-baseline-8b"), labels("holdout60-ours-8b")
    cut = {r["id"]: r.get("truncated", False) for r in jsonl(RUNS / "holdout60-baseline-8b" / "trace.jsonl")}
    out = {}
    for task in "AB":
        ids = sorted(i for i in gold if i.endswith("-" + task))
        out[task] = {
            "cases": len(ids),
            "correct": {"baseline": sum(base[i] == gold[i] for i in ids), "ours": sum(ours[i] == gold[i] for i in ids)},
            "ci95": {"baseline": boot(ids, lambda s: macro_f1(s, base, gold)),
                     "ours": boot(ids, lambda s: macro_f1(s, ours, gold)),
                     "ours_minus_baseline": boot(ids, lambda s: macro_f1(s, ours, gold) - macro_f1(s, base, gold))},
        }
    ids = [i for i in gold if i.endswith("-A")]
    for name, flag in (("not_cut", False), ("cut", True)):
        sub = [i for i in ids if cut[i] == flag]
        out["A"][name] = {"cases": len(sub), "baseline": sum(base[i] == gold[i] for i in sub),
                          "ours": sum(ours[i] == gold[i] for i in sub)}
    return out


def pairs(work, official_eval):
    cases = jsonl(work / "holdout" / "cases.jsonl")
    gold = by_id(jsonl(work / "holdout" / "expected-labels.jsonl"))
    pred = by_id(jsonl(RUNS / "holdout-ours-8b" / "predictions.jsonl"))
    groups = defaultdict(list)
    for c in cases:
        src = c["booklet"]["language"] if c["id"].endswith("-A") else c["reference"]["language"]
        groups[(c["id"][-1], f"{src}->{c['claim']['language']}")].append(c)
    out = {"A": {}, "B": {}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for (task, pair), cs in sorted(groups.items()):
            row = {"cases": len(cs), "correct": sum(pred[c["id"]]["label"] == gold[c["id"]]["label"] for c in cs)}
            if task == "A":  # evidence per pair, scored by the official evaluator on the subset
                for name, rows in (("p", [pred[c["id"]] for c in cs]), ("e", [gold[c["id"]] for c in cs])):
                    (tmp / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                       encoding="utf-8")
                subprocess.run([sys.executable, str(official_eval), "--predictions", str(tmp / "p.jsonl"),
                                "--expected", str(tmp / "e.jsonl"), "--json", str(tmp / "s.json")],
                               check=True, capture_output=True)
                ev = json.loads((tmp / "s.json").read_text(encoding="utf-8"))["tasks"]["A"]["evidence"]
                row["evidence_found"], row["evidence_cases"] = ev["found"], ev["cases"]
            out[task][pair] = row
    return out


def topk(work):
    gold = {r["id"]: r["label"] for r in jsonl(work / "devA150b" / "expected-labels.jsonl")}
    ids = sorted(gold)
    p = {k: labels(f"devA150b-ours-8b-{k}") for k in ("k4", "k6", "k8")}
    return {"cases": len(ids),
            "correct": {k: sum(v[i] == gold[i] for i in ids) for k, v in p.items()},
            "ci95_k8_minus_k4": boot(ids, lambda s: macro_f1(s, p["k8"], gold) - macro_f1(s, p["k4"], gold)),
            "ci95_k4_minus_k6": boot(ids, lambda s: macro_f1(s, p["k4"], gold) - macro_f1(s, p["k6"], gold))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--official-eval", type=Path, required=True)
    args = ap.parse_args()
    analysis = {"bootstrap": {"resamples": RESAMPLES, "seed": SEED, "interval": "95% percentile, paired"},
                "holdout60": holdout60(args.work), "holdout_pairs": pairs(args.work, args.official_eval),
                "dev_topk": topk(args.work)}
    target = ROOT / "eval" / "analysis.json"
    target.write_text(json.dumps(analysis, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
