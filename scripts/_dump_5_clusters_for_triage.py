"""One-shot dump of the 5 worst (expected, predicted) failure clusters for manual triage.

Reads test_failures.csv (strict) and test_failures_lenient.csv, intersects,
prints all 73 rows with rail context, marks which are multi_aspect_undercredit
(auto-classified as such because lenient already rescued them).

Outputs `dataset/_audits/_5cluster_for_triage.csv` as input to manual triage.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(
    r"C:\Users\FSOS\Downloads\complaint classifier-20260501T130400Z-3-001\complaint classifier"
)
AUDITS = PROJECT_ROOT / "dataset" / "_audits"

CLUSTERS = [
    ("دقة الطلب", "جودة الطعام"),
    ("النظافة", "عامة"),
    ("السعر والقيمة", "جودة الطعام"),
    ("النظافة", "جودة الطعام"),
    ("التوصيل", "جودة الطعام"),
]


def main() -> int:
    strict = pd.read_csv(AUDITS / "test_failures.csv", encoding="utf-8-sig")
    lenient = pd.read_csv(AUDITS / "test_failures_lenient.csv", encoding="utf-8-sig")

    in5 = []
    for exp, pred in CLUSTERS:
        sub = strict[(strict["expected_label"] == exp) & (strict["predicted_top"] == pred)]
        in5.append(sub)
    all5 = pd.concat(in5, ignore_index=True)

    lenient_ids = set(lenient["id"])
    all5["is_undercredit"] = ~all5["id"].isin(lenient_ids)
    out_path = AUDITS / "_5cluster_for_triage.csv"
    all5.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Total 5-cluster rows: {len(all5)}")
    print(f"Undercredit (auto-pass via top-2 rail): {int(all5['is_undercredit'].sum())}")
    print(f"Need manual triage: {int((~all5['is_undercredit']).sum())}")
    print(f"Wrote {out_path}")
    print()

    # Print each row compactly
    for _, r in all5.iterrows():
        flag = " [UNDERCREDIT]" if r["is_undercredit"] else ""
        rail = str(r.get("rail_categories", "") or "")
        print(
            f"{r['id']:>5}  {r['expected_label']:>15} -> "
            f"{r['predicted_top']:>15}  rail={rail:<40}  text={r['text']}{flag}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
