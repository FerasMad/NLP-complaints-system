"""
Tests for the data pipeline: split integrity, no leakage, label_map consistency.
"""
import csv
import json

import pandas as pd
import pytest


def test_label_map_class_count(label_map_path, schema_version):
    """Class count must match the declared schema_version.

    8class:           8 categories, no ambience
    9class_ambience:  9 categories, ambience included (v5 experiment)
    """
    with open(label_map_path, encoding="utf-8") as f:
        m = json.load(f)
    expected = {"8class": 8, "9class_ambience": 9}.get(schema_version)
    assert expected is not None, f"unknown schema_version: {schema_version!r}"
    assert len(m) == expected, (
        f"schema_version={schema_version!r} expects {expected} classes, got {len(m)}"
    )


def test_label_map_ambience_presence(label_map_path, schema_version):
    """Ambience must be absent in 8class and present in 9class_ambience.

    This replaces the old `test_label_map_no_ambiance` which permanently
    blocked ambience. The v3 → v4 ambience drop was correct for the
    production schema but should not block the v5 experiment.
    """
    with open(label_map_path, encoding="utf-8") as f:
        m = json.load(f)
    if schema_version == "8class":
        assert "الجو والمكان" not in m, (
            "ambience must NOT be present in the 8-class production schema "
            "(was dropped in v4 due to label noise)"
        )
    elif schema_version == "9class_ambience":
        assert "الجو والمكان" in m, (
            "ambience MUST be present in the 9-class experimental schema"
        )
    else:
        pytest.fail(f"unknown schema_version: {schema_version!r}")


def test_label_map_indices_contiguous(label_map_path):
    with open(label_map_path, encoding="utf-8") as f:
        m = json.load(f)
    indices = sorted(m.values())
    assert indices == list(range(len(m))), f"label indices not contiguous: {indices}"


def test_train_val_test_files_exist(project_root):
    for f in ["train.csv", "val.csv", "test.csv", "label_map.json"]:
        p = project_root / "data" / "processed" / f
        assert p.exists(), f"missing {p}"


def test_train_only_sources_not_in_val_test(project_root):
    """The crucial leakage test: synthetic/augmented sources must not appear in val or test.

    `ambience_synthetic` is the v5 source name — added here so that when the
    experimental data lands, the existing leakage gate covers it without
    requiring a second test file.
    """
    train_only = {
        "synthetic", "augmented_bt", "chatgpt_synthetic",
        "pseudo_labeled", "eda_augmented", "ambience_synthetic",
    }
    for split in ["val", "test"]:
        path = project_root / "data" / "processed" / f"{split}.csv"
        df = pd.read_csv(path)
        leaked = df[df["source"].isin(train_only)]
        assert len(leaked) == 0, f"{split}.csv has {len(leaked)} leaked train-only rows"


def test_val_test_disjoint(project_root):
    """No row should appear in both val and test."""
    val = pd.read_csv(project_root / "data" / "processed" / "val.csv")
    test = pd.read_csv(project_root / "data" / "processed" / "test.csv")
    overlap = set(val["text"]) & set(test["text"])
    assert len(overlap) == 0, f"{len(overlap)} text(s) appear in both val and test"


def test_train_disjoint_from_val_test(project_root):
    """No real text should appear in train AND (val or test) — sample check."""
    train = pd.read_csv(project_root / "data" / "processed" / "train.csv")
    val = pd.read_csv(project_root / "data" / "processed" / "val.csv")
    test = pd.read_csv(project_root / "data" / "processed" / "test.csv")
    val_test_texts = set(val["text"]) | set(test["text"])
    # Check: only REAL train rows (synthetic could legitimately match a synthetic-looking real)
    train_real = train[~train["source"].isin({"synthetic", "augmented_bt", "chatgpt_synthetic", "pseudo_labeled", "eda_augmented"})]
    overlap = set(train_real["text"]) & val_test_texts
    assert len(overlap) == 0, f"{len(overlap)} real text(s) appear in train AND (val or test)"


def test_label_column_matches_label_map(label_map_path, project_root):
    with open(label_map_path, encoding="utf-8") as f:
        m = json.load(f)
    for split in ["train", "val", "test"]:
        df = pd.read_csv(project_root / "data" / "processed" / f"{split}.csv")
        for _, r in df.head(50).iterrows():
            assert m[r["category"]] == r["label"], f"{split}: row label mismatch for category {r['category']}"
