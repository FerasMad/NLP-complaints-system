# Contributing Guide

## Branch Naming

Each member works on their own branch:

```
feras/data-collection
rima/model-training
lana/data-processing
khowla/data-processing
meshal/api-integration
mohammed/model-training
```

Create your branch from `main`:
```bash
git checkout main
git pull origin main
git checkout -b lana/data-processing
```

Open a Pull Request to `main` when your phase is complete.

---

## Current Status — Phase 0

The project is in Phase 0 (setup and onboarding). Data collection and all downstream tasks will be coordinated by the team lead. No code beyond environment setup and the synthetic reference script is expected at this stage.

---

## What NOT to Commit

- Raw or processed data files (`data/`)
- Trained model files (`models/*.pkl`)
- Environment files (`.env`)
- Python cache (`__pycache__/`)

See `.gitignore` for the full list.
