"""Run the 500-row production test fixture against the live HuggingFace Space.

Calls FerasMad/arabic-complaints-classifier via gradio_client, parses the HTML
output, scores each row by attack_type, and writes:
  - dataset/_audits/test_500_raw_predictions.csv  (raw, traceability)
  - dataset/_audits/test_failures.csv             (failures only, with reason)
  - dataset/_audits/test_run_results.md           (the report)

Usage: py scripts\\run_test_500_against_live_space.py

Reads the fixture as utf-8-sig because pandas / Windows produce BOMs.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
from gradio_client import Client

# ---- Config ----------------------------------------------------------------

SPACE_ID = "FerasMad/arabic-complaints-classifier"
API_NAME = "/predict"
DEFAULT_THROTTLE_S = 0.4  # per the brief — 1500ms is too slow, 400ms ≈ 3-4 min
SLOW_THROTTLE_S = 1.0

PROJECT_ROOT = Path(
    r"C:\Users\FSOS\Downloads\complaint classifier-20260501T130400Z-3-001\complaint classifier"
)
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "production_test_500.csv"
AUDITS_DIR = PROJECT_ROOT / "dataset" / "_audits"
RAW_CSV = AUDITS_DIR / "test_500_raw_predictions.csv"
FAIL_CSV = AUDITS_DIR / "test_failures.csv"
REPORT_MD = AUDITS_DIR / "test_run_results.md"

# 8 production categories as they appear on the rail
PRODUCTION_CATEGORIES = {
    "التوصيل",
    "السعر والقيمة",
    "النظافة",
    "جودة الطعام",
    "خدمة الموظفين",
    "دقة الطلب",
    "عامة",
    "وقت الانتظار",
}

# ---- HTML parsing ----------------------------------------------------------

# Abstain message uses `class="result-message"` (see render_message in app.py).
ABSTAIN_RE = re.compile(r'class="result-message"', re.IGNORECASE)
# General fallback uses `class="general-fallback"` with headline "شكوى عامة".
GENERAL_FALLBACK_RE = re.compile(r'class="general-fallback"', re.IGNORECASE)
# Multi-aspect badge
MULTI_BADGE_RE = re.compile(r'class="multi-aspect-badge"', re.IGNORECASE)
# Rail row category extractor: each row has class="result-row result-rank-{top|other}"
# and a `<div class="result-cat">CATEGORY</div>` inside.
RAIL_ROW_RE = re.compile(
    r'<div class="result-row\s+result-rank-(top|other)">'
    r'.*?<div class="result-cat">\s*(.*?)\s*</div>',
    re.IGNORECASE | re.DOTALL,
)


def parse_html(html: str) -> dict:
    """Return a structured dict describing what the Space rendered."""
    if not html:
        return {
            "abstain_fired": False,
            "general_fallback_fired": False,
            "multi_badge_present": False,
            "rail_categories": [],
            "predicted_top": "",
        }

    abstain = bool(ABSTAIN_RE.search(html))
    general_fb = bool(GENERAL_FALLBACK_RE.search(html))
    multi = bool(MULTI_BADGE_RE.search(html))

    rows = RAIL_ROW_RE.findall(html)
    # Filter to known production categories — defensive against fallback rows
    rail_categories = [cat for _rank, cat in rows if cat in PRODUCTION_CATEGORIES]

    if abstain:
        predicted_top = ""  # abstain — no category prediction
    elif general_fb:
        predicted_top = "عامة"
    elif rail_categories:
        predicted_top = rail_categories[0]
    else:
        predicted_top = ""

    return {
        "abstain_fired": abstain,
        "general_fallback_fired": general_fb,
        "multi_badge_present": multi,
        "rail_categories": rail_categories,
        "predicted_top": predicted_top,
    }


# ---- Scoring ---------------------------------------------------------------


def score_row(row: dict, parsed: dict) -> tuple[bool, str]:
    """Return (passed, reason_if_failed) per the per-attack-type rules in the brief."""
    expected = (row.get("expected_label") or "").strip()
    attack = (row.get("attack_type") or "").strip()

    top = parsed["predicted_top"]
    rail = parsed["rail_categories"]
    abstain = parsed["abstain_fired"]
    general_fb = parsed["general_fallback_fired"]

    # Expected-label overrides (abstain / no_complaint / multi)
    if expected == "abstain":
        return (abstain, "" if abstain else f"expected abstain; got top={top!r} rail={rail!r}")
    if expected == "no_complaint":
        ok = abstain or general_fb
        return (ok, "" if ok else f"expected abstain or general fallback; got top={top!r} rail={rail!r}")
    if expected == "multi":
        ok = len(rail) >= 2
        return (ok, "" if ok else f"expected >=2 rail categories; got rail={rail!r}")

    # Otherwise score by attack_type
    if attack == "clean":
        ok = top == expected
        return (ok, "" if ok else f"clean: expected={expected!r} got={top!r} rail={rail!r}")
    if attack == "multi_aspect":
        cond_two = len(rail) >= 2
        cond_dominant = expected in rail
        ok = cond_two and cond_dominant
        if ok:
            return (True, "")
        reasons = []
        if not cond_two:
            reasons.append(f"rail<2 (rail={rail!r})")
        if not cond_dominant:
            reasons.append(f"dominant {expected!r} not in rail")
        return (False, "multi_aspect: " + "; ".join(reasons))
    if attack == "mixed_dialect":
        ok = top == expected
        return (ok, "" if ok else f"mixed_dialect: expected={expected!r} got={top!r}")
    if attack == "sarcasm":
        ok = top == expected
        return (ok, "" if ok else f"sarcasm: expected={expected!r} got={top!r}")
    if attack == "negation":
        ok = top == expected
        return (ok, "" if ok else f"negation: expected={expected!r} got={top!r}")
    if attack == "ood":
        ok = abstain or general_fb
        return (ok, "" if ok else f"ood: expected abstain/general; got top={top!r} rail={rail!r}")
    if attack == "very_short":
        ok = abstain
        return (ok, "" if ok else f"very_short: expected abstain; got top={top!r} rail={rail!r}")
    if attack == "long":
        ok = top == expected
        return (ok, "" if ok else f"long: expected={expected!r} got={top!r}")

    ok = top == expected
    return (ok, "" if ok else f"unknown attack_type={attack!r}: expected={expected!r} got={top!r}")


# ---- Driver ----------------------------------------------------------------


def call_with_retry(client: Client, text: str, *, max_retries: int = 3) -> tuple[str, str]:
    """Returns (html, error). If error is non-empty, html is ''."""
    last_err = ""
    for attempt in range(max_retries):
        try:
            out = client.predict(text=text, api_name=API_NAME)
            if out is None:
                last_err = "none_response"
            elif isinstance(out, str):
                return (out, "")
            else:
                return (str(out), "")
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.0 * (attempt + 1))  # mild backoff
    return ("", last_err)


def run(limit: int | None = None, throttle: float = DEFAULT_THROTTLE_S) -> None:
    AUDITS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FIXTURE_PATH, encoding="utf-8-sig")
    if limit:
        df = df.head(limit)
    total = len(df)

    sys.stderr.write(f"Loaded {total} rows from {FIXTURE_PATH.name}\n")
    sys.stderr.write(f"Connecting to Space {SPACE_ID} ...\n")
    sys.stderr.flush()

    client = Client(SPACE_ID)

    raw_rows: list[dict] = []
    fail_rows: list[dict] = []
    error_count = 0

    started = time.time()

    for i, row in enumerate(df.itertuples(index=False), start=1):
        rowd = row._asdict()
        text = rowd.get("text") or ""

        html, err = call_with_retry(client, text)
        if err:
            error_count += 1
            parsed = parse_html("")
            outcome = "error"
            fail_reason = f"infra: {err}"
            passed = False
        elif not html.strip():
            error_count += 1
            parsed = parse_html("")
            outcome = "empty_response"
            fail_reason = "empty_response"
            passed = False
        else:
            parsed = parse_html(html)
            passed, fail_reason = score_row(rowd, parsed)
            outcome = "pass" if passed else "fail"

        raw_record = {
            "id": rowd.get("id", ""),
            "text": text,
            "expected_label": rowd.get("expected_label", ""),
            "difficulty": rowd.get("difficulty", ""),
            "attack_type": rowd.get("attack_type", ""),
            "notes": rowd.get("notes", ""),
            "predicted_top": parsed["predicted_top"],
            "rail_categories": " | ".join(parsed["rail_categories"]),
            "multi_badge_present": parsed["multi_badge_present"],
            "abstain_fired": parsed["abstain_fired"],
            "general_fallback_fired": parsed["general_fallback_fired"],
            "outcome": outcome,
            "failure_reason": fail_reason if not passed else "",
            "raw_html_length": len(html),
        }
        raw_rows.append(raw_record)
        if not passed:
            fail_rows.append(raw_record)

        if i % 50 == 0 or i == total:
            elapsed = time.time() - started
            sys.stderr.write(
                f"  [{i:>4}/{total}] {outcome:>14}  elapsed={elapsed:.1f}s  errs={error_count}\n"
            )
            sys.stderr.flush()

        time.sleep(throttle)

    # ---- Write outputs ----------------------------------------------------
    fieldnames = list(raw_rows[0].keys()) if raw_rows else []

    with RAW_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(raw_rows)

    with FAIL_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(fail_rows)

    # ---- Build report -----------------------------------------------------
    pass_count = sum(1 for r in raw_rows if r["outcome"] == "pass")
    fail_count = sum(1 for r in raw_rows if r["outcome"] == "fail")
    err_count = sum(1 for r in raw_rows if r["outcome"] in ("error", "empty_response"))
    pass_rate = pass_count / total if total else 0.0

    # Per-category pass rate on CLEAN attack
    cat_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [pass, fail]
    for r in raw_rows:
        if r["attack_type"] != "clean":
            continue
        if r["outcome"] in ("error", "empty_response"):
            continue
        cat = r["expected_label"]
        idx = 0 if r["outcome"] == "pass" else 1
        cat_stats[cat][idx] += 1

    # Per-attack-type pass rate
    attack_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # [pass, fail, err]
    for r in raw_rows:
        at = r["attack_type"]
        if r["outcome"] == "pass":
            attack_stats[at][0] += 1
        elif r["outcome"] == "fail":
            attack_stats[at][1] += 1
        else:
            attack_stats[at][2] += 1

    # Most common (expected, predicted) failure pairs
    pair_counter: Counter = Counter()
    for r in fail_rows:
        if r["outcome"] != "fail":
            continue
        pair_counter[(r["expected_label"], r["predicted_top"] or "<abstain/none>")] += 1
    top_pairs = pair_counter.most_common(5)

    # Notable failures — up to 10, prefer distinct attack/expected/predicted combos
    notable: list[dict] = []
    seen_pairs: set = set()
    for r in fail_rows:
        if r["outcome"] != "fail":
            continue
        key = (r["attack_type"], r["expected_label"], r["predicted_top"])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        notable.append(r)
        if len(notable) >= 10:
            break

    # Ship verdict
    if pass_rate >= 0.85:
        verdict = "ship"
        verdict_line = (
            f"**Ship verdict:** ship — pass rate {pass_rate * 100:.1f}% ≥ 85% threshold."
        )
    elif pass_rate >= 0.70:
        verdict = "ship-with-known-gaps"
        verdict_line = (
            f"**Ship verdict:** ship-with-known-gaps — pass rate {pass_rate * 100:.1f}% "
            f"between 70% and 85%; document failure modes."
        )
    else:
        verdict = "iterate"
        verdict_line = (
            f"**Ship verdict:** iterate — pass rate {pass_rate * 100:.1f}% < 70%; not ready."
        )

    # ---- Compose markdown -------------------------------------------------
    lines: list[str] = []
    lines.append("# Test Run Results — Live Space (commit 6f91ad5)")
    lines.append("")
    lines.append(
        "**Fixture:** tests/fixtures/production_test_500.csv "
        "(500 hand-written Saudi/Gulf rows, 0% overlap with training set)"
    )
    lines.append(f"**Date:** {date.today().isoformat()}")
    lines.append(f"**Space:** https://huggingface.co/spaces/{SPACE_ID}")
    lines.append(f"**Total rows:** {total}")
    lines.append(f"**Passed:** {pass_count} / {total} ({pass_rate * 100:.1f}%)")
    lines.append(f"**Failed:** {fail_count}")
    lines.append(f"**Errors:** {err_count}")
    lines.append("")
    lines.append("## Per-category pass rate (clean attack only)")
    lines.append("")
    lines.append("| Category | Pass | Fail | Rate |")
    lines.append("|---|---:|---:|---:|")

    def _rate(stats: list[int]) -> float:
        p, fl = stats
        return p / (p + fl) if (p + fl) else 0.0

    for cat in sorted(cat_stats.keys(), key=lambda c: _rate(cat_stats[c])):
        p, fl = cat_stats[cat]
        rate = _rate(cat_stats[cat])
        lines.append(f"| {cat} | {p} | {fl} | {rate * 100:.1f}% |")
    lines.append("")
    lines.append("## Per-attack-type pass rate")
    lines.append("")
    lines.append("| Attack type | Pass | Fail | Errors | Rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for at in sorted(
        attack_stats.keys(),
        key=lambda a: (-(attack_stats[a][0] + attack_stats[a][1] + attack_stats[a][2]), a),
    ):
        p, fl, er = attack_stats[at]
        denom = p + fl
        rate = (p / denom) if denom else 0.0
        lines.append(f"| {at} | {p} | {fl} | {er} | {rate * 100:.1f}% |")
    lines.append("")
    lines.append("## Failure analysis")
    lines.append("")
    lines.append("### Most common failure pattern (top 5 expected→predicted pairs)")
    lines.append("")
    if not top_pairs:
        lines.append("_No failures._")
    else:
        lines.append("| Expected | Predicted | Count |")
        lines.append("|---|---|---:|")
        for (exp, pred), cnt in top_pairs:
            lines.append(f"| {exp} | {pred} | {cnt} |")
    lines.append("")
    lines.append("### Notable failures")
    lines.append("")
    if not notable:
        lines.append("_No failures to highlight._")
    else:
        for r in notable:
            txt = r["text"]
            if len(txt) > 200:
                txt = txt[:200] + "…"
            lines.append(f"- **id={r['id']}** [{r['attack_type']}/{r['difficulty']}]")
            lines.append(f"  - text: `{txt}`")
            lines.append(f"  - expected: `{r['expected_label']}`")
            pred_display = r["predicted_top"] or "(abstain/none)"
            lines.append(
                f"  - predicted_top: `{pred_display}`  rail: `{r['rail_categories']}`  "
                f"multi_badge: {r['multi_badge_present']}  abstain: {r['abstain_fired']}  "
                f"general_fb: {r['general_fallback_fired']}"
            )
            lines.append(f"  - reason: {r['failure_reason']}")
    lines.append("")
    lines.append("## Ship verdict")
    lines.append("")
    lines.append(verdict_line)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Generated by `scripts/run_test_500_against_live_space.py`.")
    lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    # ---- Console summary --------------------------------------------------
    sys.stderr.write("\n=== SUMMARY ===\n")
    sys.stderr.write(f"Total rows:   {total}\n")
    sys.stderr.write(f"Passed:       {pass_count} ({pass_rate * 100:.1f}%)\n")
    sys.stderr.write(f"Failed:       {fail_count}\n")
    sys.stderr.write(f"Errors:       {err_count}\n")
    sys.stderr.write(f"Verdict:      {verdict}\n")
    sys.stderr.write(f"Report:       {REPORT_MD}\n")
    sys.stderr.write(f"Failures CSV: {FAIL_CSV}\n")
    sys.stderr.write(f"Raw preds:    {RAW_CSV}\n")
    sys.stderr.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Cap rows (for smoke test)")
    parser.add_argument("--throttle", type=float, default=DEFAULT_THROTTLE_S)
    parser.add_argument("--slow", action="store_true", help="Use slower 1.0s throttle")
    args = parser.parse_args()
    throttle = SLOW_THROTTLE_S if args.slow else args.throttle
    run(limit=args.limit, throttle=throttle)


if __name__ == "__main__":
    main()
