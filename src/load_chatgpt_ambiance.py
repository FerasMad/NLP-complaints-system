"""
Load ChatGPT-generated ambiance complaints, clean, dedupe, and append to
data/text/complaints_labeled.csv as source="chatgpt_synthetic".

Treated as train-only (split_dataset.py keeps source!="play_store"/"production"/"res1" out of val/test).

Input: data/raw/chatgpt_ambiance.txt (one complaint per line)

Usage:
    py src/load_chatgpt_ambiance.py
"""
import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "raw" / "chatgpt_ambiance.txt"
LABELED = ROOT / "data" / "text" / "complaints_labeled.csv"
TARGET_CATEGORY = "الجو والمكان"

# Lana's cleaning (vendored)
TASHKEEL = re.compile(r"[ً-ٰٟؐ-ؚ]")
NON_ARABIC = re.compile(r"[^؀-ۿa-zA-Z0-9٠-٩\s]")
WHITESPACE = re.compile(r"\s+")
ARABIC_CHAR = re.compile(r"[؀-ۿ]")


def clean(text):
    if not text:
        return ""
    t = TASHKEEL.sub("", text)
    t = t.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"}))
    t = NON_ARABIC.sub(" ", t)
    return WHITESPACE.sub(" ", t).strip().lower()


def is_arabic(text):
    if not text or len(text) < 5:
        return False
    arabic_chars = len(ARABIC_CHAR.findall(text))
    return arabic_chars / max(len(text), 1) > 0.3


def existing_texts():
    seen = set()
    with open(LABELED, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            seen.add(r.get("text", "").strip())
    return seen


def main():
    if not INPUT.exists():
        print(f"ERROR: {INPUT} not found")
        sys.exit(1)

    seen = existing_texts()
    print(f"Existing texts in labeled file: {len(seen)}")

    raw_lines = INPUT.read_text(encoding="utf-8").splitlines()
    print(f"Raw lines from {INPUT}: {len(raw_lines)}")

    rows = []
    drops = Counter()
    for line in raw_lines:
        line = line.strip()
        if not line:
            drops["blank"] += 1
            continue
        if not is_arabic(line):
            drops["not_arabic"] += 1
            continue
        cleaned = clean(line)
        if not cleaned or len(cleaned) < 5:
            drops["too_short"] += 1
            continue
        if cleaned in seen:
            drops["duplicate"] += 1
            continue
        seen.add(cleaned)
        rows.append({
            "text": cleaned,
            "category": TARGET_CATEGORY,
            "priority": "متوسطة",
            "source": "chatgpt_synthetic",
        })

    print(f"Kept: {len(rows)}")
    print(f"Drops: {dict(drops)}")

    if not rows:
        print("Nothing to add. Exiting.")
        return

    with open(LABELED, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category", "priority", "source"])
        writer.writerows(rows)
    print(f"Appended {len(rows)} ChatGPT ambiance rows -> {LABELED}")


if __name__ == "__main__":
    main()
