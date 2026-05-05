"""Generalized label-noise audit for any category in val + test.

Same heuristic as src/audit_ambiance_eval.py but:
  - Works for any category via --category flag
  - Outputs a per-category audit CSV the user can review
  - Summary table across all 8 categories at end (when run with --all)

Usage:
    py src/audit_category.py --category "عامة"
    py src/audit_category.py --all
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

# Reuse keyword sets + classification heuristic from the ambience audit
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_ambience_eval import KW, classify, count_hits

ROOT = Path(__file__).resolve().parent.parent
VAL_CSV = ROOT / "data" / "processed" / "val.csv"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
OUT_DIR = ROOT / "data" / "processed"


def audit_category(category: str) -> dict:
    """Return summary dict + write per-category audit CSV."""
    rows = []
    for split, path in [("val", VAL_CSV), ("test", TEST_CSV)]:
        df = pd.read_csv(path)
        sub = df[df.category == category].copy()
        for _, r in sub.iterrows():
            text = str(r["text"])
            hits = {cat: count_hits(text, kws) for cat, kws in KW.items()}
            classification, notes = classify_for_target(text, hits, target=category)
            dom = max(((c, h) for c, h in hits.items() if h > 0), key=lambda x: x[1], default=("none", 0))
            rows.append({
                "source_split": split,
                "text": text,
                **{f"hits_{c}": h for c, h in hits.items()},
                "dominant_category": dom[0],
                "dominant_hits": dom[1],
                "my_classification": classification,
                "notes_auto": notes,
            })

    df_out = pd.DataFrame(rows)
    summary = dict(df_out["my_classification"].value_counts())
    out_csv = OUT_DIR / f"audit_{category.replace(' ', '_')}.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return {
        "category": category,
        "rows": len(df_out),
        "summary": summary,
        "csv": out_csv,
    }


def classify_for_target(text: str, hits: dict, target: str) -> tuple[str, str]:
    """Classify whether the row's gold label `target` matches the keyword evidence."""
    target_hits = hits.get(target, 0)
    others = {k: v for k, v in hits.items() if k != target and v > 0}

    if target_hits == 0 and not others:
        return "ambiguous", "no clear keywords"
    if target_hits == 0:
        dom = max(others.items(), key=lambda x: x[1])
        return "probably_mislabeled", f"no {target} keywords; dominant: {dom[0]} ({dom[1]} hits)"
    if target_hits >= 2 and not others:
        return "clean_target", f"{target_hits} {target} keywords, no competing category"
    if target_hits >= 2 and others:
        max_other = max(others.values())
        if max_other >= target_hits:
            dom = max(others.items(), key=lambda x: x[1])
            return "multi_aspect", f"{target}={target_hits}, competing {dom[0]}={dom[1]}"
        return "multi_aspect", f"{target} dominant ({target_hits}) but also: {others}"
    if target_hits == 1 and not others:
        return "clean_target", "1 keyword, no others"
    if target_hits == 1 and others:
        max_other = max(others.values())
        if max_other > target_hits:
            dom = max(others.items(), key=lambda x: x[1])
            return "probably_mislabeled", f"weak (1), stronger: {dom[0]} ({dom[1]})"
        return "ambiguous", f"weak (1), comparable: {others}"
    return "ambiguous", f"hits: {hits}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", type=str, default=None)
    ap.add_argument("--all", action="store_true", help="Audit all 8 categories + summary")
    args = ap.parse_args()

    if args.all:
        all_cats = ["التوصيل", "السعر والقيمة", "النظافة", "جودة الطعام", "خدمة الموظفين", "دقة الطلب", "عامة", "وقت الانتظار"]
        results = []
        for cat in all_cats:
            r = audit_category(cat)
            results.append(r)
            print(f"\n{cat}: {r['rows']} samples")
            for k, v in r["summary"].items():
                print(f"  {k}: {v}")

        # Cross-category summary
        out_path = OUT_DIR / "audit_all_categories_summary.txt"
        lines = ["# Per-category label-noise audit (val + test)", ""]
        lines.append(f"{'Category':<20s} {'Rows':>6s} {'clean':>8s} {'multi':>8s} {'mis-lab':>9s} {'ambig':>7s} {'mis-lab%':>9s}")
        for r in results:
            s = r["summary"]
            clean = s.get("clean_target", 0)
            multi = s.get("multi_aspect", 0)
            mis = s.get("probably_mislabeled", 0)
            amb = s.get("ambiguous", 0)
            mispct = mis / max(r["rows"], 1) * 100
            line = f"{r['category']:<20s} {r['rows']:>6d} {clean:>8d} {multi:>8d} {mis:>9d} {amb:>7d} {mispct:>8.1f}%"
            lines.append(line)
        lines.append("")
        lines.append("**Interpretation:**")
        lines.append("- mis-lab% > 30%: category may have label-noise issues like ambiance had. Worth manual review.")
        lines.append("- multi% > 50%: multi-aspect dominance — consider multi-label training.")
        lines.append("- High clean% + low mis-lab%: trustworthy gold labels.")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nSummary -> {out_path}")
        for line in lines:
            print(line)
    elif args.category:
        r = audit_category(args.category)
        print(f"\n{r['category']}: {r['rows']} samples")
        for k, v in r["summary"].items():
            print(f"  {k}: {v}")
        print(f"\nDetailed CSV -> {r['csv']}")
    else:
        ap.error("specify --category <name> or --all")


if __name__ == "__main__":
    main()
