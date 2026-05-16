"""Write the manually-classified triage CSV.

Reads `dataset/_audits/_5cluster_for_triage.csv` (73 rows from the 5 worst
failure clusters), applies the hand-classification table below, and writes
`dataset/_audits/test_failures_triaged.csv` with a new `failure_class` column.

Classifications:
  real_miss               — model genuinely wrong, expected label is fine
  ambiguous               — both labels defensible (e.g., wet bag = delivery OR cleanliness)
  fixture_typo            — expected was wrong; predicted is actually right
  multi_aspect_undercredit — expected IS in the rail but not top-1 (auto: is_undercredit=True)
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(
    r"C:\Users\FSOS\Downloads\complaint classifier-20260501T130400Z-3-001\complaint classifier"
)
AUDITS = PROJECT_ROOT / "dataset" / "_audits"

# Hand-classification for the 73 cluster rows.
# `multi_aspect_undercredit` is auto-applied from is_undercredit; not listed here.
# `fixture_typo` would override the expected label too; none found in this sample.
AMBIGUOUS_IDS = {
    # Cluster 2 (hygiene → general): foreign-object/food-safety overlap
    "T129",  # nail in burger — hygiene or food quality
    "T135",  # broken plates — hygiene or general damage
    "T139",  # small stone in rice cup — hygiene or food quality
    "T150",  # plastic bits in foul plate — hygiene or food quality
    # Cluster 3 (price → food): tax/promotion/tipping spillover
    "T060",  # promotion mismatch — price or staff service
    "T077",  # tax applied to discounted meal — pricing accuracy
    "T086",  # premium membership not honored — price or service
    "T091",  # price mismatch from menu — price or order accuracy
    "T097",  # forced 20% tip — price or staff service
    # Cluster 4 (hygiene → food): dirty plate / residue / cold-buffet overlap
    "T102",  # plate has stains from previous food — hygiene or food quality
    "T120",  # cup arrived with lemon slice from another customer — both
    "T127",  # food left on table from yesterday — both
    "T138",  # plate dripping from incomplete washing — both
    "T147",  # buffet food falling and being put back — both
    # Cluster 5 (delivery → food): cold delivery
    "T041",  # always arrives cold — food quality or delivery quality
}


def classify(row: pd.Series) -> str:
    if row.get("is_undercredit") in {True, "True", "TRUE", 1, "1"}:
        return "multi_aspect_undercredit"
    if row["id"] in AMBIGUOUS_IDS:
        return "ambiguous"
    return "real_miss"


def main() -> int:
    src = AUDITS / "_5cluster_for_triage.csv"
    df = pd.read_csv(src, encoding="utf-8-sig")
    df["failure_class"] = df.apply(classify, axis=1)

    out = AUDITS / "test_failures_triaged.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Wrote {out} ({len(df)} rows)")
    print()
    counts = df["failure_class"].value_counts()
    print("Classification breakdown (73 rows from 5 worst clusters):")
    for k, n in counts.items():
        print(f"  {k:>25}  {n}")
    print()
    # Per-cluster
    df["cluster"] = df["expected_label"] + " -> " + df["predicted_top"]
    pivot = df.pivot_table(
        index="cluster", columns="failure_class", values="id", aggfunc="count", fill_value=0
    )
    print("Per-cluster breakdown:")
    print(pivot.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
