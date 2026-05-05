from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from data import import_arama_ambience, import_hf_datasets_for_ambience, import_kaggle_or_local_reviews  # noqa: E402

FIXTURE = _ROOT / "tests" / "fixtures" / "ambience_mini_reviews.csv"
EXPECTED_FILES = [
    "ambience_candidates.csv",
    "hard_negative_candidates.csv",
    "review_needed.csv",
    "source_summary.json",
]
REQUIRED_COLUMNS = {
    "id",
    "text",
    "category",
    "source",
    "weak_label",
    "contains_ambience_keyword",
    "ambience_subtype",
    "risk_keyword",
    "is_hard_negative",
    "label_confidence",
    "review_status",
    "notes",
    "split",
}


def _assert_output_contract(output_dir: Path) -> dict:
    for filename in EXPECTED_FILES:
        assert (output_dir / filename).exists(), filename
    for filename in EXPECTED_FILES[:3]:
        df = pd.read_csv(output_dir / filename, encoding="utf-8-sig")
        assert REQUIRED_COLUMNS <= set(df.columns)
    summary = json.loads((output_dir / "source_summary.json").read_text(encoding="utf-8"))
    assert summary["row_count_raw"] == 20
    assert summary["row_count_arabic"] == 18
    assert summary["row_count_arabic"] < summary["row_count_raw"]
    return summary


def test_arama_importer_writes_expected_files(tmp_path, monkeypatch):
    output_dir = tmp_path / "arama"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import_arama_ambience.py",
            "--input",
            str(FIXTURE),
            "--text-col",
            "text",
            "--aspect-col",
            "aspect",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert import_arama_ambience.main() == 0
    summary = _assert_output_contract(output_dir)
    assert summary["ambience_candidates_count"] > 0
    assert summary["hard_negative_candidates_count"] > 0


def test_local_importer_writes_expected_files(tmp_path, monkeypatch):
    output_dir = tmp_path / "local"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import_kaggle_or_local_reviews.py",
            "--input",
            str(FIXTURE),
            "--source-name",
            "kaggle_arabic_reviews",
            "--text-col",
            "text",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert import_kaggle_or_local_reviews.main() == 0
    summary = _assert_output_contract(output_dir)
    assert summary["ambience_candidates_count"] > 0
    assert summary["review_needed_count"] > 0


def test_hf_importer_writes_expected_files(tmp_path, monkeypatch):
    output_dir = tmp_path / "hf"
    fixture_df = pd.read_csv(FIXTURE, encoding="utf-8")
    fixture_df = fixture_df.rename(columns={"text": "review"})
    fixture_df["polarity"] = "negative"
    fixture_df["restaurant_id"] = "r1"

    monkeypatch.setattr(import_hf_datasets_for_ambience, "_dataset_to_frame", lambda *_: fixture_df)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import_hf_datasets_for_ambience.py",
            "--dataset",
            "fixture/hf",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert import_hf_datasets_for_ambience.main() == 0
    summary = _assert_output_contract(output_dir)
    assert summary["ambience_candidates_count"] > 0
    assert summary["source_name"] == "huggingface"
