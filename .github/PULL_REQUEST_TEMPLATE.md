## Summary
<!-- 1-3 bullets on what this changes and why -->

## Type of change
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (API contract or schema)
- [ ] Model retraining / data pipeline change
- [ ] Documentation only
- [ ] Refactor / housekeeping

## Test plan
- [ ] `pytest tests/test_data_pipeline.py` passes
- [ ] `pytest -m "not gpu"` passes locally
- [ ] If model changed: `python src/eval_ensemble_on_test.py ...` shows accuracy >= 92%
- [ ] If API changed: `python -m app.api` starts cleanly + manual curl works
- [ ] If deps changed: `pip install -e .` works in a fresh venv

## Checklist
- [ ] If schema/contract changed: README.md updated
- [ ] No model weight files committed (use HF Hub instead)
- [ ] No secrets / tokens committed (HF_TOKEN, OPENAI_API_KEY, etc.)
- [ ] Type hints + docstrings added on public functions
- [ ] No commented-out code or print-debug leftovers

## Reviewer notes
<!-- Anything specific you want the reviewer to focus on, or risks to flag -->
