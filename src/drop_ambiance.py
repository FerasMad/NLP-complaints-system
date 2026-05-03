"""
Drop the ambiance category entirely.

The val/test labels for ambiance are ~99% noise (per audit_ambiance_eval.py:
171 samples → only 2 truly clean ambiance, 47% mislabeled, 40% multi-aspect).
The category is unrecoverable without rebuilding the eval set from scratch.

This script:
  1. Backs up data/text/complaints_labeled.csv -> complaints_labeled.with_ambiance.csv
  2. Removes all rows with category = "الجو والمكان"
  3. Writes a clean 8-category CSV
  4. Re-splits via split_dataset.py (which auto-builds new label_map.json)

After running: retrain models for 8-class.

Usage:
    py src/drop_ambiance.py
"""
from __future__ import annotations

import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
LABELED = ROOT / "data" / "text" / "complaints_labeled.csv"
BACKUP = ROOT / "data" / "text" / "complaints_labeled.with_ambiance.csv"
DROP_CATEGORY = "الجو والمكان"


def main():
    if not LABELED.exists():
        print(f"ERROR: {LABELED} not found")
        sys.exit(1)

    if not BACKUP.exists():
        print(f"Backing up {LABELED} -> {BACKUP}")
        shutil.copy2(LABELED, BACKUP)
    else:
        print(f"Backup already exists at {BACKUP} (keeping)")

    # Read all rows
    with open(LABELED, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    print(f"Loaded {len(rows)} rows")

    before_counts = Counter(r["category"] for r in rows)
    kept = [r for r in rows if r.get("category") != DROP_CATEGORY]
    dropped = len(rows) - len(kept)
    print(f"Dropped {dropped} rows with category={DROP_CATEGORY}")
    print(f"Remaining: {len(kept)} rows")

    # Write back to LABELED
    with open(LABELED, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    print(f"Wrote {LABELED}")

    after_counts = Counter(r["category"] for r in kept)
    print("\nCategory counts:")
    print(f"{'category':<25s} before  after")
    for cat in sorted(set(before_counts) | set(after_counts)):
        b = before_counts.get(cat, 0)
        a = after_counts.get(cat, 0)
        marker = " <-- DROPPED" if cat == DROP_CATEGORY else ""
        print(f"{cat:<25s} {b:>6d}  {a:>6d}{marker}")

    print()
    print("Now run: py src/split_dataset.py")
    print("  (will produce 8-class train/val/test + new label_map.json)")


if __name__ == "__main__":
    main()
