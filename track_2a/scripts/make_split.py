"""Freeze a development / held-out split of the public OST dataset.

The split is by vote date (one date = one booklet in three languages), so the held-out set
contains only votes the system never saw during development. This mimics the private
evaluation set, which may contain unseen booklets. Run once; the result is committed.
"""

import argparse
import hashlib
import json
from pathlib import Path

HOLDOUT_DATES = 5  # out of 20 vote dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True, help="v1.1.jsonl from OSTswiss/MNLIoverSwissVotingBooklets")
    ap.add_argument("--out", type=Path, default=Path("data/split.json"))
    args = ap.parse_args()
    raw = args.dataset.read_bytes()
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    dates = sorted({r["booklet_publish_date"] for r in rows})
    ranked = sorted(dates, key=lambda d: hashlib.sha256(f"hackapertus-split-v1:{d}".encode()).hexdigest())
    holdout = sorted(ranked[:HOLDOUT_DATES])
    split = {
        "dataset": "OSTswiss/MNLIoverSwissVotingBooklets v1.1.jsonl",
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "rule": "sha256('hackapertus-split-v1:'+date), lowest 5 dates are held out",
        "holdout_dates": holdout,
        "dev_dates": [d for d in dates if d not in holdout],
        "rows": {"dev": [i for i, r in enumerate(rows) if r["booklet_publish_date"] not in holdout],
                 "holdout": [i for i, r in enumerate(rows) if r["booklet_publish_date"] in holdout]},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split, indent=1), encoding="utf-8")
    print(f"dev rows {len(split['rows']['dev'])}, holdout rows {len(split['rows']['holdout'])}, holdout dates {holdout}")


if __name__ == "__main__":
    main()
