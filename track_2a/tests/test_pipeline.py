import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from booklet_check import pipeline  # noqa: E402
from booklet_check.llm import Apertus  # noqa: E402
from booklet_check.pdf_index import build_index, split_page  # noqa: E402

EXAMPLES = ROOT / "data" / "examples"


def test_parse_label():
    assert pipeline.parse_label("Entailment") == 0
    assert pipeline.parse_label("contradiction.") == 2
    assert pipeline.parse_label(" neutral\n") == 1
    assert pipeline.parse_label("I am not sure") is None


def test_chunks_are_verbatim_slices():
    page = "Erste Zeile des Textes.\nZweite Zeile mit 1 000 Franken.\n\nNeuer Absatz hier, lang genug um zu bleiben."
    for piece in split_page(page, target=20, hard_max=60):
        assert piece in page


def test_evidence_pages_point_at_the_quoted_text():
    idx = build_index(EXAMPLES / "booklets" / "2024_09_22_de.pdf")
    for c in idx.chunks[:50]:
        assert c.text in idx.pages[c.page - 1]


def test_dry_mode_writes_valid_predictions(tmp_path, monkeypatch):
    for k in ("BASE_URL", "API_KEY", "LLM_BASE_URL", "LLM_API_KEY", "APERTUS_BASE_URL", "APERTUS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from booklet_check.__main__ import main

    out = tmp_path / "pred.jsonl"
    main(["--input", str(EXAMPLES / "cases.jsonl"), "--output", str(out), "--workers", "1"])
    preds = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    cases = [json.loads(l) for l in (EXAMPLES / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [p["id"] for p in preds] == [c["id"] for c in cases]
    for p in preds:
        assert p["label_name"] == pipeline.LABELS[p["label"]]
        assert set(p["metrics"]) == {"input_tokens", "output_tokens", "inference_time_ms"}
        if p["label"] != 1 and p["id"].endswith("-A"):
            assert p["evidence"] and all(isinstance(e["page"], int) and e["page"] >= 1 for e in p["evidence"])
    assert Apertus().dry
