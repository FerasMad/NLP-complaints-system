import sys
import io
import pandas as pd
from datasets import load_dataset

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# -- OCLAR -> complaints_labeled.csv ───────────────────────────────────────────
print("Loading OCLAR (CC BY 4.0)...")
oclar = load_dataset("community-datasets/oclar", split="train")
df_oclar = oclar.to_pandas()
df_neg = df_oclar[df_oclar["rating"].isin([1, 2])].copy()
df_neg["source_label"] = df_neg["rating"].apply(lambda r: f"oclar_{r}star")
df_neg = df_neg.rename(columns={"review": "text"})[["source_label", "text"]]
df_neg = df_neg.dropna(subset=["text"])

labeled_path = r"C:\Users\Feras\Desktop\labeled_text\complaints_labeled.csv"
df_neg.to_csv(labeled_path, mode="a", header=False, index=False, encoding="utf-8-sig")
print(f"  -> Appended {len(df_neg)} rows to complaints_labeled.csv")

# -- ar_res_reviews -> complaints_unlabeled.csv ────────────────────────────────
# Note: dataset uses 'text' column (not 'review'); polarity 1 = positive
print("Loading ar_res_reviews...")
ar_rev = load_dataset("hadyelsahar/ar_res_reviews", split="train")
df_ar = ar_rev.to_pandas()
df_pos = df_ar[df_ar["polarity"] == 1][["text"]].copy()
df_pos = df_pos.dropna(subset=["text"])

unlabeled_path = r"C:\Users\Feras\Desktop\unlabeled_text\complaints_unlabeled.csv"
df_pos.to_csv(unlabeled_path, mode="a", header=False, index=False, encoding="utf-8-sig")
print(f"  -> Appended {len(df_pos)} rows to complaints_unlabeled.csv")

print("Done.")
