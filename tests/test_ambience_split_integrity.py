"""Integrity tests for the schema-aware splitter (`src/split_dataset.py`).

These tests exercise the splitter on a tiny in-memory dataset rather than
the production data, so they run in CI without requiring `data/processed/`
to exist. They guard the rules that matter most:

  * `ambience_synthetic` and `ambience_eda_augmented` rows must NOT leak
    into val or test (the v3 ambience class died from train-leak issues
    among other things).
  * The 9-class schema must keep الجو والمكان in the label map.
  * The 8-class schema must drop ambience rows silently (back-compat).
  * The split manifest must record the active schema_version.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
SPLITTER = _ROOT / "src" / "split_dataset.py"


def _make_tiny_dataset(tmp_path: Path, include_ambience: bool) -> Path:
    """Write a tiny labeled CSV with each production category + optional ambience."""
    rows = []
    # Each category gets enough rows that 70/15/15 stratified split works
    categories = [
        "التوصيل", "السعر والقيمة", "النظافة", "جودة الطعام",
        "خدمة الموظفين", "دقة الطلب", "عامة", "وقت الانتظار",
    ]
    if include_ambience:
        categories.append("الجو والمكان")

    for cat in categories:
        for i in range(20):  # 20 real rows per category
            rows.append({
                "text": f"{cat} test row {i}",
                "category": cat,
                "label": "",
                "priority": "1",
                "source": "manual",
            })
        # 5 train-only rows per category (these MUST stay out of val/test)
        for i in range(5):
            src = "ambience_synthetic" if cat == "الجو والمكان" else "synthetic"
            rows.append({
                "text": f"{cat} synth row {i}",
                "category": cat,
                "label": "",
                "priority": "1",
                "source": src,
            })

    csv_path = tmp_path / "complaints_labeled.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "category", "label", "priority", "source"])
        w.writeheader()
        w.writerows(rows)
    return csv_path


def _run_splitter(csv_path: Path, out_dir: Path, schema_version: str):
    """Run the splitter as a subprocess; raise if it fails."""
    result = subprocess.run(
        [
            sys.executable, str(SPLITTER),
            "--input", str(csv_path),
            "--output-dir", str(out_dir),
            "--schema-version", schema_version,
        ],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, f"splitter failed: {result.stderr}"


def test_8class_drops_ambience_silently(tmp_path):
    """When the 8-class splitter runs on data with ambience rows, those rows are dropped."""
    csv_path = _make_tiny_dataset(tmp_path, include_ambience=True)
    out_dir = tmp_path / "out_8class"
    _run_splitter(csv_path, out_dir, "8class")

    with open(out_dir / "label_map.json", encoding="utf-8") as f:
        m = json.load(f)
    assert len(m) == 8, f"expected 8 classes, got {len(m)}"
    assert "الجو والمكان" not in m

    with open(out_dir / "split_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["schema_version"] == "8class"


def test_9class_keeps_ambience(tmp_path):
    """When the 9-class splitter runs, ambience must be in the label map."""
    csv_path = _make_tiny_dataset(tmp_path, include_ambience=True)
    out_dir = tmp_path / "out_9class"
    _run_splitter(csv_path, out_dir, "9class_ambience")

    with open(out_dir / "label_map.json", encoding="utf-8") as f:
        m = json.load(f)
    assert len(m) == 9, f"expected 9 classes, got {len(m)}"
    assert "الجو والمكان" in m

    with open(out_dir / "split_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["schema_version"] == "9class_ambience"
    assert manifest["num_classes"] == 9


def test_ambience_synthetic_stays_in_train_only(tmp_path):
    """ambience_synthetic and ambience_eda_augmented must never appear in val/test."""
    import pandas as pd
    csv_path = _make_tiny_dataset(tmp_path, include_ambience=True)
    out_dir = tmp_path / "out_leakage"
    _run_splitter(csv_path, out_dir, "9class_ambience")

    train_only = {"synthetic", "ambience_synthetic", "ambience_eda_augmented"}
    for split_name in ("val", "test"):
        df = pd.read_csv(out_dir / f"{split_name}.csv")
        leaked = df[df["source"].isin(train_only)]
        assert leaked.empty, (
            f"{split_name}.csv leaked {len(leaked)} train-only rows:\n"
            + "\n".join(f"  {r['source']}: {r['text']}" for _, r in leaked.iterrows())
        )


def test_split_manifest_contains_source_counts(tmp_path):
    csv_path = _make_tiny_dataset(tmp_path, include_ambience=True)
    out_dir = tmp_path / "out_manifest"
    _run_splitter(csv_path, out_dir, "9class_ambience")

    with open(out_dir / "split_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    counts = manifest["source_counts"]
    assert counts.get("manual", 0) > 0
    assert counts.get("synthetic", 0) > 0
    assert counts.get("ambience_synthetic", 0) > 0


def test_no_duplicate_text_across_splits(tmp_path):
    import pandas as pd
    csv_path = _make_tiny_dataset(tmp_path, include_ambience=True)
    out_dir = tmp_path / "out_dedup"
    _run_splitter(csv_path, out_dir, "9class_ambience")

    train = pd.read_csv(out_dir / "train.csv")
    val = pd.read_csv(out_dir / "val.csv")
    test = pd.read_csv(out_dir / "test.csv")

    # val ∩ test must be empty
    assert not (set(val["text"]) & set(test["text"])), "val ∩ test contains duplicates"
    # real rows in train must not appear in val or test
    train_real = train[~train["source"].isin({"synthetic", "ambience_synthetic", "ambience_eda_augmented"})]
    overlap = set(train_real["text"]) & (set(val["text"]) | set(test["text"]))
    assert not overlap, f"real train rows leak into val/test: {overlap}"
