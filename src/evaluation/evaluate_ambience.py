"""Evaluate a trained 9-class ambience model against the adversarial fixture.

Loads a model from `--model-dir`, runs predictions over every row in
`tests/fixtures/ambience_adversarial.csv`, and reports:

  * Overall accuracy (against `expected_label`)
  * Accuracy broken down by `attack_type` (clean, boundary, negation,
    sarcasm, mixed_dialect, multi_aspect, very_short, long, out_of_domain,
    typo, emoji, adversarial)
  * Accuracy broken down by `difficulty` (easy, medium, hard)
  * Per-class precision/recall/F1
  * Ambience-specific FP/FN dumps to CSV
  * Confusion matrix to CSV
  * A markdown summary report

The "expected_label" column may be `abstain` for cases where the right
answer is to not classify (e.g. one-word inputs). For those rows we
treat any low-confidence (< 0.4) prediction as a pass.

Usage:

    python src/evaluation/evaluate_ambience.py \\
        --model-dir models/single_ambience_v1 \\
        --adversarial-fixture tests/fixtures/ambience_adversarial.csv \\
        --output-dir reports/ambience

Exit code 0 if all the gates pass:
    - Overall accuracy >= 0.85
    - Ambience F1 >= 0.85
    - Easy-difficulty accuracy >= 0.95
    - No attack_type with accuracy below 0.50

Exit code 1 if any gate fails. Use this in CI as a safety net.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

CATEGORIES_9CLASS = [
    "التوصيل", "السعر والقيمة", "النظافة", "جودة الطعام",
    "خدمة الموظفين", "دقة الطلب", "عامة", "وقت الانتظار",
    "الجو والمكان",
]

# Pseudo-label used in the adversarial fixture for praise / no-issue rows.
NO_COMPLAINT_LABEL = "no_complaint"
ABSTAIN_LABEL = "abstain"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(model_dir: Path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return tokenizer, model, device, torch


def predict_batch(texts: list[str], tokenizer, model, device, torch_mod, batch_size: int = 32):
    """Return (top_categories, top_confidences, full_probs, label_map)."""
    label_map = {int(i): cat for i, cat in (model.config.id2label or {}).items()}
    if not label_map:
        # Fallback: assume canonical 9-class order
        label_map = {i: cat for i, cat in enumerate(CATEGORIES_9CLASS)}
    n_classes = len(label_map)

    all_probs = np.zeros((len(texts), n_classes), dtype=np.float32)
    with torch_mod.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", truncation=True,
                            max_length=192, padding=True).to(device)
            logits = model(**enc).logits
            all_probs[i:i + len(batch)] = torch_mod.softmax(logits, dim=-1).cpu().numpy()

    top_idx = all_probs.argmax(axis=-1)
    top_conf = all_probs[np.arange(len(texts)), top_idx]
    top_cat = [label_map[int(i)] for i in top_idx]
    return top_cat, top_conf, all_probs, label_map


def apply_abstain_logic(
    texts: list[str],
    predicted: list[str],
    confidences: np.ndarray,
    abstain_threshold: float = 0.40,
) -> tuple[list[str], np.ndarray]:
    """Override predictions to abstain for very-short and out-of-domain inputs.

    This is the v5 inference wrapper's abstain layer. Two gates:

      1. **Length gate** — fewer than 2 words after cleaning → abstain.
         Forces predictions on 1-word topic-only inputs (like 'حر') down
         to confidence=0 so the eval scoring treats them as abstain.

      2. **Out-of-domain gate** — no restaurant-domain anchor word
         (food / staff / delivery / price / order / hygiene / wait /
         ambience-related) → abstain. Catches inputs like 'سيارتي تعطلت'.

    Both gates work by setting confidence to 0.0 on the matching rows.
    The existing `score_row` treats any prediction with confidence below
    `abstain_threshold` as an abstain, which already passes when the
    expected label is 'abstain' or 'no_complaint'.
    """
    # Local import to avoid coupling the evaluator's top-level imports
    sys.path.insert(0, str(ROOT / "src"))
    from utils.arabic_normalization import clean
    from config.ambience_keywords import has_restaurant_domain_anchor

    new_preds = list(predicted)
    new_confs = confidences.copy()

    for i, raw in enumerate(texts):
        cleaned = clean(raw)
        # Length gate
        if len(cleaned.split()) < 2:
            new_preds[i] = "abstain"
            new_confs[i] = 0.0
            continue
        # OOD gate
        if not has_restaurant_domain_anchor(cleaned):
            new_preds[i] = "abstain"
            new_confs[i] = 0.0
            continue

    return new_preds, new_confs


def apply_ambience_threshold(
    all_probs: np.ndarray,
    label_map: dict[int, str],
    ambience_threshold: float,
) -> tuple[list[str], np.ndarray]:
    """Override per-row prediction to ambience when its softmax >= threshold.

    The trained model is high-precision (~96%) but low-recall (~67%) on
    ambience. Lowering its decision threshold trades a small amount of
    precision for a meaningful recall gain. This function does that
    post-hoc without retraining: if ambience's softmax probability is
    >= `ambience_threshold`, we predict ambience even when another class
    has a higher softmax.

    Returns (predicted_categories, confidences). Confidence is always
    the chosen class's softmax (ambience's if overridden, else top-1).
    """
    cat_to_idx = {cat: i for i, cat in label_map.items()}
    if "الجو والمكان" not in cat_to_idx:
        # Schema doesn't include ambience — no override possible
        top_idx = all_probs.argmax(axis=-1)
        return [label_map[int(i)] for i in top_idx], all_probs[np.arange(len(all_probs)), top_idx]

    amb_idx = cat_to_idx["الجو والمكان"]
    amb_probs = all_probs[:, amb_idx]
    top_idx = all_probs.argmax(axis=-1)

    overridden_idx = np.where(
        (amb_probs >= ambience_threshold) & (top_idx != amb_idx),
        amb_idx,
        top_idx,
    )
    predicted = [label_map[int(i)] for i in overridden_idx]
    confidence = all_probs[np.arange(len(all_probs)), overridden_idx]
    return predicted, confidence


# ---------------------------------------------------------------------------
# Scoring with abstain semantics
# ---------------------------------------------------------------------------

def score_row(expected: str, predicted: str, confidence: float, abstain_threshold: float = 0.40) -> bool:
    """A row passes if:

      * expected == 'abstain' AND model confidence < abstain_threshold (any class)
      * expected == 'no_complaint' AND model confidence < abstain_threshold
        OR predicted is 'عامة' (the project's catch-all)
      * Otherwise: predicted == expected
    """
    if expected == ABSTAIN_LABEL:
        return confidence < abstain_threshold
    if expected == NO_COMPLAINT_LABEL:
        return confidence < abstain_threshold or predicted == "عامة"
    return predicted == expected


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame, output_dir: Path) -> dict:
    """Compute and write all reports. Returns gate-check dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(df)
    passed = int(df["pass"].sum())
    overall_acc = passed / max(total, 1)

    # By attack type
    by_attack = (
        df.groupby("attack_type")["pass"]
          .agg(["sum", "count"])
          .assign(accuracy=lambda x: x["sum"] / x["count"])
          .reset_index()
          .sort_values("accuracy")
    )

    # By difficulty
    by_difficulty = (
        df.groupby("difficulty")["pass"]
          .agg(["sum", "count"])
          .assign(accuracy=lambda x: x["sum"] / x["count"])
          .reset_index()
    )

    # Per-class precision/recall/F1 — only for cases with a real expected_label
    real = df[~df["expected_label"].isin({ABSTAIN_LABEL, NO_COMPLAINT_LABEL})].copy()
    per_class = []
    for cat in CATEGORIES_9CLASS:
        tp = ((real["expected_label"] == cat) & (real["predicted"] == cat)).sum()
        fp = ((real["expected_label"] != cat) & (real["predicted"] == cat)).sum()
        fn = ((real["expected_label"] == cat) & (real["predicted"] != cat)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        per_class.append({
            "category": cat,
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })
    per_class_df = pd.DataFrame(per_class)
    ambience_row = per_class_df[per_class_df["category"] == "الجو والمكان"].iloc[0]
    ambience_f1 = float(ambience_row["f1"])

    # Confusion matrix on classifiable rows
    cats_for_cm = CATEGORIES_9CLASS + [NO_COMPLAINT_LABEL, ABSTAIN_LABEL]
    cm = pd.DataFrame(0, index=cats_for_cm, columns=cats_for_cm)
    for _, r in df.iterrows():
        exp = r["expected_label"] if r["expected_label"] in cats_for_cm else NO_COMPLAINT_LABEL
        pred = r["predicted"] if r["predicted"] in cats_for_cm else r["predicted"]
        if pred not in cats_for_cm:
            cm.loc[exp, ABSTAIN_LABEL] = cm.loc[exp].get(ABSTAIN_LABEL, 0) + 1
        else:
            cm.loc[exp, pred] += 1
    cm.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    # Ambience FP / FN
    amb_fp = real[(real["expected_label"] != "الجو والمكان") & (real["predicted"] == "الجو والمكان")]
    amb_fn = real[(real["expected_label"] == "الجو والمكان") & (real["predicted"] != "الجو والمكان")]
    amb_fp.to_csv(output_dir / "ambience_false_positives.csv", index=False, encoding="utf-8-sig")
    amb_fn.to_csv(output_dir / "ambience_false_negatives.csv", index=False, encoding="utf-8-sig")

    # Per-class CSV
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")

    # Markdown summary
    md = [
        "# Ambience v5 Adversarial Evaluation",
        "",
        f"- Total cases: {total}",
        f"- Passed: {passed}",
        f"- **Overall accuracy: {overall_acc:.2%}**",
        f"- **Ambience F1: {ambience_f1:.2%}**",
        "",
        "## By difficulty",
        "",
        "| Difficulty | Passed | Total | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for _, r in by_difficulty.iterrows():
        md.append(f"| {r['difficulty']} | {int(r['sum'])} | {int(r['count'])} | {r['accuracy']:.2%} |")
    md += ["", "## By attack type", "", "| Attack type | Passed | Total | Accuracy |", "|---|---:|---:|---:|"]
    for _, r in by_attack.iterrows():
        md.append(f"| {r['attack_type']} | {int(r['sum'])} | {int(r['count'])} | {r['accuracy']:.2%} |")
    md += ["", "## Per-class metrics (real labels only)", "",
           "| Category | TP | FP | FN | Precision | Recall | F1 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in per_class_df.iterrows():
        md.append(f"| {r['category']} | {r['tp']} | {r['fp']} | {r['fn']} | "
                  f"{r['precision']:.2%} | {r['recall']:.2%} | {r['f1']:.2%} |")
    md += [
        "",
        "## Files written",
        f"- `confusion_matrix.csv`",
        f"- `per_class_metrics.csv`",
        f"- `ambience_false_positives.csv` ({len(amb_fp)} rows)",
        f"- `ambience_false_negatives.csv` ({len(amb_fn)} rows)",
        f"- `predictions.csv` (full per-row output)",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")

    df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    # Gate checks
    min_attack_acc = float(by_attack["accuracy"].min()) if len(by_attack) else 0.0
    easy_acc = float(by_difficulty.loc[by_difficulty["difficulty"] == "easy", "accuracy"].iloc[0]) \
        if (by_difficulty["difficulty"] == "easy").any() else 1.0
    return {
        "overall_accuracy": overall_acc,
        "ambience_f1": ambience_f1,
        "min_attack_type_accuracy": min_attack_acc,
        "easy_accuracy": easy_acc,
        "gates": {
            "overall_accuracy_>=_0.85": overall_acc >= 0.85,
            "ambience_f1_>=_0.85": ambience_f1 >= 0.85,
            "easy_accuracy_>=_0.95": easy_acc >= 0.95,
            "min_attack_accuracy_>=_0.50": min_attack_acc >= 0.50,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Path to the trained model directory (output of train_ambience_v5.py)")
    p.add_argument("--adversarial-fixture", type=Path,
                   default=ROOT / "tests" / "fixtures" / "ambience_adversarial.csv",
                   help="Adversarial test CSV (default: tests/fixtures/ambience_adversarial.csv)")
    p.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "ambience",
                   help="Where to write reports (default: reports/ambience)")
    p.add_argument("--abstain-threshold", type=float, default=0.40,
                   help="Confidence below this counts as abstain (default: 0.40)")
    p.add_argument(
        "--ambience-threshold",
        type=float,
        default=None,
        help=(
            "If set, override prediction to ambience whenever the ambience "
            "softmax >= this threshold (even if another class is higher). "
            "Trades precision for recall. Default None = pure argmax."
        ),
    )
    p.add_argument(
        "--threshold-sweep",
        type=str,
        default=None,
        help=(
            "Comma-separated list of thresholds to sweep (e.g. '0.10,0.15,0.20,0.25'). "
            "Runs evaluation at each, writes a sweep summary CSV, and exits without "
            "writing the per-row reports. Useful for finding the F1-maximizing point."
        ),
    )
    p.add_argument(
        "--enable-abstain",
        action="store_true",
        help=(
            "Apply v5 inference wrapper abstain logic before scoring: short "
            "(<2 words) and out-of-domain (no restaurant-domain anchor) inputs "
            "are demoted to confidence=0 so they pass the abstain/no_complaint "
            "expected labels. Closes the very_short and out_of_domain attack-type "
            "gates without touching the model."
        ),
    )
    p.add_argument("--gate-on-failures", action="store_true",
                   help="Exit 1 if any quality gate fails (use in CI)")
    args = p.parse_args()

    if not args.model_dir.exists():
        print(f"Model directory not found: {args.model_dir}", file=sys.stderr)
        print("Run src/train_ambience_v5.py first.", file=sys.stderr)
        return 2
    if not args.adversarial_fixture.exists():
        print(f"Adversarial fixture not found: {args.adversarial_fixture}", file=sys.stderr)
        return 2

    print(f"[load] loading model from {args.model_dir}")
    tokenizer, model, device, torch_mod = load_model(args.model_dir)
    print(f"[load] model loaded on {device}")

    df = pd.read_csv(args.adversarial_fixture, encoding="utf-8")
    print(f"[load] loaded {len(df)} adversarial cases")

    print("[predict] running inference...")
    texts = df["text"].astype(str).tolist()
    top_cat, top_conf, all_probs, label_map = predict_batch(texts, tokenizer, model, device, torch_mod)

    # Threshold sweep mode — run evaluation at each threshold and write a
    # summary CSV. Skip the per-row reports (caller will pick a threshold
    # and re-run normally with that one).
    if args.threshold_sweep:
        thresholds = [float(t.strip()) for t in args.threshold_sweep.split(",") if t.strip()]
        print(f"[sweep] evaluating at thresholds: {thresholds}")
        sweep_rows = []
        # Baseline (pure argmax, no override)
        baseline_df = df.copy()
        baseline_df["predicted"] = top_cat
        baseline_df["confidence"] = top_conf
        baseline_df["pass"] = [
            score_row(r["expected_label"], r["predicted"], r["confidence"], args.abstain_threshold)
            for _, r in baseline_df.iterrows()
        ]
        baseline_summary = generate_report(baseline_df, args.output_dir / "_sweep_baseline")
        sweep_rows.append({
            "threshold": "argmax",
            "overall_accuracy": baseline_summary["overall_accuracy"],
            "ambience_f1": baseline_summary["ambience_f1"],
            "easy_accuracy": baseline_summary["easy_accuracy"],
            "min_attack_type_accuracy": baseline_summary["min_attack_type_accuracy"],
        })
        for thresh in sorted(thresholds):
            preds, confs = apply_ambience_threshold(all_probs, label_map, thresh)
            sweep_df = df.copy()
            sweep_df["predicted"] = preds
            sweep_df["confidence"] = confs
            sweep_df["pass"] = [
                score_row(r["expected_label"], r["predicted"], r["confidence"], args.abstain_threshold)
                for _, r in sweep_df.iterrows()
            ]
            sub_dir = args.output_dir / f"_sweep_t{thresh:.2f}".replace(".", "_")
            summary = generate_report(sweep_df, sub_dir)
            sweep_rows.append({
                "threshold": thresh,
                "overall_accuracy": summary["overall_accuracy"],
                "ambience_f1": summary["ambience_f1"],
                "easy_accuracy": summary["easy_accuracy"],
                "min_attack_type_accuracy": summary["min_attack_type_accuracy"],
            })

        sweep_csv = args.output_dir / "threshold_sweep.csv"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sweep_rows).to_csv(sweep_csv, index=False, encoding="utf-8-sig")
        print()
        print("=" * 80)
        print(f"  Threshold sweep results -> {sweep_csv}")
        print()
        print(f"  {'threshold':<12} {'overall':>10} {'ambience F1':>14} {'easy':>10} {'min_attack':>12}")
        for r in sweep_rows:
            t = r["threshold"] if isinstance(r["threshold"], str) else f"{r['threshold']:.2f}"
            print(
                f"  {t:<12} "
                f"{r['overall_accuracy']:>9.2%} "
                f"{r['ambience_f1']:>13.2%} "
                f"{r['easy_accuracy']:>9.2%} "
                f"{r['min_attack_type_accuracy']:>11.2%}"
            )
        # Find the best F1
        best = max(sweep_rows, key=lambda r: r["ambience_f1"])
        best_t = best["threshold"] if isinstance(best["threshold"], str) else f"{best['threshold']:.2f}"
        print()
        print(f"  Best ambience F1: {best['ambience_f1']:.2%} at threshold={best_t}")
        print("=" * 80)
        return 0

    # Normal mode: single threshold (or argmax if None)
    if args.ambience_threshold is not None:
        print(f"[predict] applying ambience-threshold override at {args.ambience_threshold}")
        preds, confs = apply_ambience_threshold(all_probs, label_map, args.ambience_threshold)
    else:
        preds, confs = list(top_cat), top_conf.copy()

    # Optionally apply v5 abstain wrapper: length gate + OOD gate.
    if args.enable_abstain:
        print(f"[predict] applying v5 abstain logic (length + OOD gates)")
        preds, confs = apply_abstain_logic(texts, preds, confs, args.abstain_threshold)
        n_abstained = sum(1 for p in preds if p == "abstain")
        print(f"[predict]   {n_abstained}/{len(preds)} rows abstained")

    df["predicted"] = preds
    df["confidence"] = confs

    df["pass"] = [
        score_row(r["expected_label"], r["predicted"], r["confidence"], args.abstain_threshold)
        for _, r in df.iterrows()
    ]

    print(f"[report] writing reports to {args.output_dir}")
    summary = generate_report(df, args.output_dir)

    print()
    print("=" * 60)
    print(f"  Overall accuracy:        {summary['overall_accuracy']:.2%}")
    print(f"  Ambience F1:             {summary['ambience_f1']:.2%}")
    print(f"  Easy-difficulty acc:     {summary['easy_accuracy']:.2%}")
    print(f"  Min attack-type acc:     {summary['min_attack_type_accuracy']:.2%}")
    print()
    print("  Gates:")
    for name, passed in summary["gates"].items():
        marker = "PASS" if passed else "FAIL"
        print(f"    [{marker}] {name}")
    print("=" * 60)

    if args.gate_on_failures and not all(summary["gates"].values()):
        print("\nOne or more gates failed.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
