# Contributing

Made by the NLP team at AI Club.

## Setup

```bash
git clone https://github.com/FerasMad/NLP-complaints-system.git
cd NLP-complaints-system
py -m venv .venv
.venv/Scripts/python -m pip install -e .
# GPU users:
# .venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Smoke test:

```bash
.venv/Scripts/python -c "from app.ensemble_inference import EnsembleClassifier; clf = EnsembleClassifier('models/ensemble_final/config.json'); print(clf.predict('الاكل بايخ ومالح'))"
```

## Layout

```
app/      # API + Gradio + EnsembleClassifier
src/      # data + training + eval scripts
tests/    # pytest suite
models/   # ensemble manifest + bakeoff dirs
data/     # labeled CSVs + canary sets
hf_space/ # HF Spaces deploy bundle
scripts/  # helpers
```

## Branches & commits

- Work on `main`. Push when results pass test set.
- Topic branches for multi-commit features: `<owner>/<task>` (e.g. `feras/distill`).
- Imperative mood, short titles ("Add per-source eval", not "Added the per-source evaluation script").
- No co-author trailers.

## PR checklist

- `pytest` passes
- If model artifacts changed: `models/ensemble_final/config.json` + README numbers updated
- No secrets committed
- No model weight files committed (use HF Hub)

## Add a training experiment

1. `src/run_bakeoff.py --models <name> --save-suffix _experiment`
2. `src/eval_on_test.py models/bakeoff/<name>_experiment_final`
3. If it beats current ensemble: add to `models/ensemble_final/config.json`, re-run `src/eval_ensemble_on_test.py`.

## Add a data source

1. Drop raw data in `data/raw/`.
2. Loader script in `src/` cleans via existing `clean()`, marks `source="<name>"`, appends to `data/text/complaints_labeled.csv`.
3. If synthetic/augmented: add to `train_only_sources` in [`src/split_dataset.py`](src/split_dataset.py) so it stays out of val/test.
4. Re-split, retrain, re-eval.

## Tests

```bash
pytest
pytest -m "not gpu"   # skip GPU tests
```

CI: [`.github/workflows/test.yml`](.github/workflows/test.yml).
