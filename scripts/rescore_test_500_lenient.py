"""Offline lenient rescore of the 500-row test predictions.

Reads `dataset/_audits/test_500_raw_predictions.csv` (already has every row's
`rail_categories` from the live-Space run) and re-applies a SOFTENED rubric:

  Strict (the original):  top-1 == expected
  Lenient (this script):  expected IN top-2 rail

Outputs two artifacts under `dataset/_audits/`:

  test_run_results_lenient.md   — strict vs lenient side-by-side report
  test_failures_lenient.csv     — failures that survive even the lenient rubric

This is a pure offline rescore. No live-Space calls. No model load.

Per the audit plan: lenient rescore is one half of "Approach C — Hybrid".
Manual triage of the worst clusters (in dataset/_audits/test_failures_triaged.csv)
is the other half. The corrected pass rate combines both.

Usage:  py scripts\\rescore_test_500_lenient.py
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(
    r"C:\Users\FSOS\Downloads\complaint classifier-20260501T130400Z-3-001\complaint classifier"
)
AUDITS_DIR = PROJECT_ROOT / "dataset" / "_audits"
RAW_CSV = AUDITS_DIR / "test_500_raw_predictions.csv"
LENIENT_REPORT = AUDITS_DIR / "test_run_results_lenient.md"
LENIENT_FAILURES_CSV = AUDITS_DIR / "test_failures_lenient.csv"

PRODUCTION_CATEGORIES = {
    "التوصيل", "السعر والقيمة", "النظافة", "جودة الطعام",
    "خدمة الموظفين", "دقة الطلب", "عامة", "وقت الانتظار",
}


def parse_rail(rail_str: str) -> list[str]:
    """Convert pipe-delimited rail string from the raw CSV to a list."""
    if pd.isna(rail_str) or not rail_str:
        return []
    return [c.strip() for c in str(rail_str).split("|") if c.strip()]


def score_lenient(row: dict) -> tuple[bool, str]:
    """Return (passed, reason).

    Lenient = (strict would pass) OR (expected in top-2 rail).

    Strict ≡ `predicted_top == expected` for normal attacks. Including the
    strict check here makes lenient a true superset — without it we'd
    regress on rows where general_fallback fired with rail=[] (e.g.,
    'عامة' general-complaint rescue cases), because the rail is empty but
    predicted_top correctly carries the rescued category.

    Special labels and abstain/very_short/ood attacks keep their original
    rules unchanged.
    """
    def _s(v) -> str:
        # Handle pandas NaN (float), None, and real strings uniformly.
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v).strip()

    expected = _s(row.get("expected_label"))
    attack = _s(row.get("attack_type"))
    rail = parse_rail(row.get("rail_categories", ""))
    predicted_top = _s(row.get("predicted_top"))
    abstain = _s(row.get("abstain_fired")).lower() == "true"
    general_fb = _s(row.get("general_fallback_fired")).lower() == "true"
    top2 = rail[:2]

    # Special expected labels — same rules as strict
    if expected == "abstain":
        return (abstain, "" if abstain else f"expected abstain; got rail={rail!r}")
    if expected == "no_complaint":
        ok = abstain or general_fb
        return (ok, "" if ok else f"expected abstain/general; got rail={rail!r}")
    if expected == "multi":
        ok = len(rail) >= 2
        return (ok, "" if ok else f"expected >=2 rail; got rail={rail!r}")

    # Per-attack-type rubric
    if attack == "ood":
        ok = abstain or general_fb
        return (ok, "" if ok else f"ood: expected abstain/general; got rail={rail!r}")
    if attack == "very_short":
        ok = abstain
        return (ok, "" if ok else f"very_short: expected abstain; got rail={rail!r}")
    if attack == "multi_aspect":
        # Same rule as strict — already lenient
        cond_two = len(rail) >= 2
        cond_present = expected in rail
        ok = cond_two and cond_present
        if ok:
            return (True, "")
        reasons = []
        if not cond_two:
            reasons.append(f"rail<2 (rail={rail!r})")
        if not cond_present:
            reasons.append(f"{expected!r} not in rail")
        return (False, "multi_aspect: " + "; ".join(reasons))

    # clean / mixed_dialect / sarcasm / negation / long →
    # lenient = strict-pass OR expected in top-2 rail
    if predicted_top == expected:
        return (True, "")
    if expected in top2:
        return (True, "")
    return (
        False,
        f"{attack}: expected={expected!r} not in top-2 rail={top2!r} "
        f"(full={rail!r}, predicted_top={predicted_top!r})",
    )


def main() -> int:
    AUDITS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    total = len(df)
    print(f"Loaded {total} rows from {RAW_CSV.name}")

    # Rescore
    raw_rows: list[dict] = []
    lenient_failures: list[dict] = []
    err_count = 0
    for row in df.itertuples(index=False):
        rowd = row._asdict()
        # Strict outcome from the original run
        strict_outcome = (rowd.get("outcome") or "").strip()
        if strict_outcome in {"error", "empty_response"}:
            err_count += 1
            lenient_outcome = "error"
            reason = rowd.get("failure_reason", "infra error")
        else:
            ok, reason = score_lenient(rowd)
            lenient_outcome = "pass" if ok else "fail"

        record = {
            "id": rowd.get("id", ""),
            "text": rowd.get("text", ""),
            "expected_label": rowd.get("expected_label", ""),
            "difficulty": rowd.get("difficulty", ""),
            "attack_type": rowd.get("attack_type", ""),
            "notes": rowd.get("notes", ""),
            "predicted_top": rowd.get("predicted_top", ""),
            "rail_categories": rowd.get("rail_categories", ""),
            "multi_badge_present": rowd.get("multi_badge_present", ""),
            "abstain_fired": rowd.get("abstain_fired", ""),
            "general_fallback_fired": rowd.get("general_fallback_fired", ""),
            "strict_outcome": strict_outcome,
            "lenient_outcome": lenient_outcome,
            "lenient_reason": "" if lenient_outcome == "pass" else reason,
        }
        raw_rows.append(record)
        if lenient_outcome == "fail":
            lenient_failures.append(record)

    # Write the lenient-failures CSV
    if raw_rows:
        with LENIENT_FAILURES_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            w.writeheader()
            w.writerows(lenient_failures)

    # ---- Aggregate ----
    strict_pass = sum(1 for r in raw_rows if r["strict_outcome"] == "pass")
    strict_fail = sum(1 for r in raw_rows if r["strict_outcome"] == "fail")
    lenient_pass = sum(1 for r in raw_rows if r["lenient_outcome"] == "pass")
    lenient_fail = sum(1 for r in raw_rows if r["lenient_outcome"] == "fail")
    rescued = sum(
        1 for r in raw_rows
        if r["strict_outcome"] == "fail" and r["lenient_outcome"] == "pass"
    )

    # Per-category (clean only, lenient)
    cat_lenient: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    cat_strict: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in raw_rows:
        if r["attack_type"] != "clean":
            continue
        if r["strict_outcome"] in {"error", "empty_response"}:
            continue
        cat = r["expected_label"]
        cat_strict[cat][0 if r["strict_outcome"] == "pass" else 1] += 1
        cat_lenient[cat][0 if r["lenient_outcome"] == "pass" else 1] += 1

    # Per-attack-type
    atk_strict: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    atk_lenient: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in raw_rows:
        at = r["attack_type"]
        s = r["strict_outcome"]
        l = r["lenient_outcome"]
        atk_strict[at][0 if s == "pass" else (1 if s == "fail" else 2)] += 1
        atk_lenient[at][0 if l == "pass" else (1 if l == "fail" else 2)] += 1

    # Failure pair counts on lenient
    lenient_pairs: Counter = Counter()
    for r in lenient_failures:
        lenient_pairs[(r["expected_label"], r["predicted_top"] or "<abstain/none>")] += 1

    # ---- Compose report ----
    pass_rate_strict = strict_pass / total if total else 0
    pass_rate_lenient = lenient_pass / total if total else 0

    def _rate(stats: list[int]) -> float:
        p, fl = stats[0], stats[1]
        return p / (p + fl) if (p + fl) else 0.0

    lines: list[str] = []
    lines.append("# Test Run Results — Lenient Rescore (Approach C, part 1)")
    lines.append("")
    lines.append(f"**Date:** {date.today().isoformat()}")
    lines.append("**Source:** `dataset/_audits/test_500_raw_predictions.csv` (live Space commit 6f91ad5)")
    lines.append("**Rubric change:** `top-1 == expected` → `expected IN rail[:2]` for clean / mixed_dialect / sarcasm / negation / long. Special rows (ood, very_short, multi, abstain, no_complaint) unchanged.")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total rows | {total} |")
    lines.append(f"| **Strict top-1 pass rate** | **{strict_pass} / {total} ({pass_rate_strict*100:.1f}%)** |")
    lines.append(f"| **Lenient (top-2 rail) pass rate** | **{lenient_pass} / {total} ({pass_rate_lenient*100:.1f}%)** |")
    lines.append(f"| Rescued by lenient rubric | {rescued} rows (strict fail → lenient pass) |")
    lines.append(f"| Errors / infrastructure failures | {err_count} |")
    lines.append("")
    lines.append("## Per-category pass rate, clean rows only (strict vs lenient)")
    lines.append("")
    lines.append("| Category | Strict pass | Strict rate | Lenient pass | Lenient rate | Lift |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cat in sorted(cat_strict.keys(), key=lambda c: _rate(cat_lenient[c])):
        s_p, s_f = cat_strict[cat]
        l_p, l_f = cat_lenient[cat]
        s_rate = _rate(cat_strict[cat])
        l_rate = _rate(cat_lenient[cat])
        lift = l_rate - s_rate
        lines.append(
            f"| {cat} | {s_p} | {s_rate*100:.1f}% | {l_p} | {l_rate*100:.1f}% | +{lift*100:.1f} pts |"
        )
    lines.append("")
    lines.append("## Per-attack-type pass rate (strict vs lenient)")
    lines.append("")
    lines.append("| Attack | Strict pass | Strict rate | Lenient pass | Lenient rate | Lift |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for at in sorted(atk_strict.keys()):
        s_p, s_f, s_e = atk_strict[at]
        l_p, l_f, l_e = atk_lenient[at]
        s_total_eff = s_p + s_f
        l_total_eff = l_p + l_f
        s_rate = (s_p / s_total_eff) if s_total_eff else 0
        l_rate = (l_p / l_total_eff) if l_total_eff else 0
        lift = l_rate - s_rate
        lines.append(
            f"| {at} | {s_p} | {s_rate*100:.1f}% | {l_p} | {l_rate*100:.1f}% | +{lift*100:.1f} pts |"
        )
    lines.append("")
    lines.append("## Top 10 surviving failure pairs (lenient rubric)")
    lines.append("")
    lines.append("These are failures where the expected label is NOT in the top-2 rail. The model genuinely missed them — they are NOT just multi-aspect undercredit cases.")
    lines.append("")
    lines.append("| Expected | Predicted top | Count |")
    lines.append("|---|---|---:|")
    for (exp, pred), n in lenient_pairs.most_common(10):
        lines.append(f"| {exp} | {pred} | {n} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"The lenient rubric rescues **{rescued} rows** ({rescued*100/total:.1f}% of total) from the strict-fail bucket. "
        f"These rows have the expected category in the rail (top-2) but not as top-1 — they're "
        f"_multi-aspect-undercredit_ cases where the aspect-driven rail already shows the user "
        f"the correct category. Counting them as passes lifts the headline from "
        f"**{pass_rate_strict*100:.1f}%** to **{pass_rate_lenient*100:.1f}%**."
    )
    lines.append("")
    lines.append(
        "**The lenient number is the upper bound. The triaged-corrected number "
        "(`dataset/_audits/test_failures_triaged.csv`) is the honest middle ground.**"
    )
    lines.append("")
    lines.append("## Ship verdict (lenient)")
    lines.append("")
    if pass_rate_lenient >= 0.85:
        v = "ship — lenient pass ≥ 85%."
    elif pass_rate_lenient >= 0.70:
        v = "ship-with-known-gaps — lenient pass between 70% and 85%."
    else:
        v = "iterate — even the lenient pass < 70%; structural improvements needed."
    lines.append(f"**Lenient verdict:** {v}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Generated by `scripts/rescore_test_500_lenient.py`. Combine with `test_failures_triaged.csv` for the corrected pass rate.")

    LENIENT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {LENIENT_REPORT}")
    print(f"Wrote {LENIENT_FAILURES_CSV} ({len(lenient_failures)} lenient failures)")
    print()
    print(f"Strict pass:   {strict_pass}/{total} ({pass_rate_strict*100:.1f}%)")
    print(f"Lenient pass:  {lenient_pass}/{total} ({pass_rate_lenient*100:.1f}%)")
    print(f"Rescued:       {rescued} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
