# Production Audit

Audit run: 2026-05-16 (Saturday) by the Production Inspector.

All four public surfaces (HF Space, HF Model, HF Dataset, GitHub) are live and consistent on the load-bearing claims (95.05% accuracy, 8-class production model, Saudi/Gulf dialect, CAMeLBERT base). Three real issues found — one inconsistency between the dataset card and the actual CSV contents, one Space hygiene leftover, and a `.gitignore` gap that could cause an accidental ~7.5 GB commit. None block ship; the dataset-card discrepancy should be fixed before showing to reviewers.

## Surface Status

| Surface | Status | Latest commit | Notes |
|---|---|---|---|
| HF Space | RUNNING (cpu-basic) | `86805fdb5432...` (Space) | Subdomain `ferasmad-arabic-complaints-classifier.hf.space` returns HTTP 200. Live `/predict` API responds for `الاكل بايخ` with HTML containing "جودة الطعام" (verified). Hosted `app.py` is byte-identical to local `hf_space/app.py` (sha256 match). |
| HF Model | OK | `5f471fd69022...` | safetensors downloadable (302 to signed URL, OK). `label_map.json` has the right 8 Arabic labels. YAML frontmatter parses. `model-index` block present with `dataset` sub-block (custom, "Arabic Restaurant Complaints (held-out test set)"). Widget examples present. |
| HF Dataset | OK with one card inaccuracy | `e92b9e825235...` | All 14 files in the README file table exist in the repo siblings list. `complaints_labeled.csv` downloads (22.4 MB) and parses to 95,391 rows. License = MIT. See Inconsistencies below for class-count claim mismatch. |
| GitHub | OK | `6f91ad5d` on `origin/main` | Latest commit is `6f91ad5` (Aspect-driven rail, 2026-05-16 14:23 UTC) — that is past `8efc572`, the expected floor. Only one branch (`main`); no stale `fix/space-bad-patterns` left. README accurately describes the 8-class model and 95.05% number. All required files exist (`MODEL_CARD.md`, `docs/SENIOR_REVIEW.md`, `docs/V5_TRAINING_RUN_1..5*.md`, `hf_space/app.py`, `dataset/README.md`). |
| Local | Mostly clean, see Warnings | `6f91ad5` (same as origin) | `git status` shows 2 modified notebooks and 3 untracked model dirs (~7.5 GB). All would normally be expected, but the `.gitignore` gap for the untracked models is a small landmine. |

## Critical issues (blockers — must fix before ship)

None.

## Warnings (should fix soon but not blockers)

- **Space has a leftover `__pycache__/app.cpython-313.pyc`** in its repo siblings list. Harmless at runtime but cosmetic noise — purge from the Space repo. (`https://huggingface.co/api/spaces/FerasMad/arabic-complaints-classifier` → siblings)
- **`.gitignore` does not cover the new pretrained model directories.** `git status` currently shows three untracked directories totaling **~7.5 GB**: `models/single_ambience_v1_pretrained/` (2.9 GB), `models/single_ambience_v1_pretrained_v2/` (2.9 GB), `models/camelbert_arabic_reviews_pretrained/` (1.7 GB). The gitignore already excludes `models/single_ambience_v1/` and `models/ensemble_ambience_v1/`, but not the `_pretrained*` variants. A stray `git add -A` would commit all 7.5 GB. Add `models/single_ambience_v1_pretrained*/` and `models/camelbert_arabic_reviews_pretrained/` to `.gitignore`.
- **Two modified notebooks not committed** — `notebooks/04_model_training.ipynb` (4 lines) and `notebooks/04b_camelbert_training.ipynb` (244 lines, net -212). These are training notebooks; the diff shows mostly output stripping. Either commit or `git checkout --` them so the working tree is clean.
- **Model `config.json` has generic `LABEL_0..LABEL_7` in `id2label`/`label2id`** instead of the Arabic category names. The Space and any caller that loads `label_map.json` will be fine, but anyone using `pipeline("text-classification", ...)` straight from `transformers` will get `LABEL_3` instead of `جودة الطعام`. Cosmetic but it would polish the model card-to-API experience.

## Inconsistencies between surfaces

- **Dataset README says `complaints_labeled.csv` is "8-class labels", but the file actually contains 9 categories.** The CSV has 95,391 rows across these category values:
  `التوصيل, الجو والمكان, السعر والقيمة, النظافة, جودة الطعام, خدمة الموظفين, دقة الطلب, عامة, وقت الانتظار`.
  The 9th class is `الجو والمكان` (ambience) — the v5 experiment class. Two README rows say "8-class" verbatim:
  - `complaints_labeled.csv` row: *"The full production training dataset. Text + 8-class labels..."*
  - `production_labels_only.csv` row: *"The 8-class production labels — the actual training labels for the deployed model."*
  Either: (a) strip ambience rows from both CSVs before re-uploading to match the "8-class" claim, or (b) update the README to say "9-class labels (8 production + 1 v5 ambience experiment)". Option (b) is faster and arguably more honest given the v5 narrative the rest of the card builds.
  Same likely applies to `production_labels_only.csv` (didn't download to check, but it's derived from the same source).
- **GitHub README category table is ID-ordered** (0–7) while MODEL_CARD's category table is F1-ordered (high to low). Not wrong — just two different orderings of the same 8 labels. Harmless, mention only because the brief asked.
- **Local main is exactly at `origin/main` (`6f91ad5`).** The brief expected `8efc572` or later; we are 4 commits past that, all on the same hf_space rendering work. Consistent.

## Cross-surface category check

8 labels match exactly across all three surfaces that use the production label set:

| Source | Labels |
|---|---|
| HF Model `label_map.json` | التوصيل, السعر والقيمة, النظافة, جودة الطعام, خدمة الموظفين, دقة الطلب, عامة, وقت الانتظار |
| `hf_space/app.py` CATEGORIES | (same 8, identical order) |
| MODEL_CARD.md categories table | (same 8) |
| GitHub README categories table | (same 8) |

The dataset CSV is the only place a 9th label appears, per the inconsistency above.

## Performance numbers cross-check

| Number | MODEL_CARD.md | GitHub README | HF Model card | docs/SENIOR_REVIEW.md |
|---|---|---|---|---|
| Test accuracy | 95.05% | 95.05% | 0.9505 (model-index) | "95% test accuracy" |
| Macro F1 | 92.03% | 92.03% | 0.9203 (model-index) | not numerically restated |
| Weighted F1 | 95.08% | not shown | 0.9508 (model-index) | not numerically restated |
| Min class F1 | 84.84% (عامة) | 84.84% | n/a | "every class > 84% F1" |

All four documents are consistent. SENIOR_REVIEW rounds rather than restating exact decimals, which is fine.

## Ship readiness verdict

Ship. All four public surfaces are healthy, the Space's live `/predict` works, the deployed model file is the same code the user can read on GitHub (verified by sha256), the model card is honest about limits, and the docs across surfaces tell a consistent story (95.05% accuracy, 8-class production, Saudi/Gulf dialect, CAMeLBERT-mix base, with the v5 ambience experiment clearly flagged as research-stage). The dataset card's "8-class" wording for the labeled CSV is the only thing a careful reviewer will catch — fix that one sentence and the surface set is publication-grade. The `.gitignore` gap and the `__pycache__` artifact in the Space are housekeeping; neither breaks anything for users.
