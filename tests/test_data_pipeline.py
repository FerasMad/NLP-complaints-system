"""
Tests for the data pipeline: split integrity, no leakage, label_map consistency.
"""
import csv
import json

import pandas as pd
import pytest


def test_label_map_is_8_classes(label_map_path):
    """Schema is 8 categories (ambiance was dropped — see README)."""
    with open(label_map_path, encoding="utf-8") as f:
        m = json.load(f)
    assert len(m) == 8, f"expected 8 classes (post-ambiance-drop), got {len(m)}"


def test_label_map_no_ambiance(label_map_path):
    with open(label_map_path, encoding="utf-8") as f:
        m = json.load(f)
    assert "الجو والمكان" not in m, "ambiance category should not be present"


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
    """The crucial leakage test: synthetic/augmented sources must not appear in val or test."""
    train_only = {"synthetic", "augmented_bt", "chatgpt_synthetic", "pseudo_labeled", "eda_augmented"}
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
