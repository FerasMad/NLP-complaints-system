# Security Policy

## Reporting a vulnerability

Email **iiiferasiii@gmail.com** with:
- Description and steps to reproduce
- Affected component (API, model loading, data pipeline, etc.)
- Your assessment of impact

Please do not open a public GitHub issue. Allow up to 7 days for a reply, and 30 days before public disclosure.

## In scope

- Code execution via crafted input
- Auth bypass on the API
- Data leakage from logs or model cache
- Dependency vulnerabilities affecting this codebase

## Out of scope

- Issues in upstream pretrained models — report to CAMeL-Lab, UBC-NLP, or aubmindlab directly
- Misclassification of legitimate text — use the GitHub issue tracker

## Operational notes for deployers

**PII.** The API ships with a built-in scrubber that masks phone/email/URL before logging (see [`app/api.py`](app/api.py)). Set short log retention. Comply with local privacy law (GDPR, Saudi PDPL, etc.).

**Model integrity.** Pin model revisions by SHA in [`models/ensemble_final/config.json`](models/ensemble_final/config.json) so upstream weight updates can't silently change behavior.

**CORS.** Default `ALLOWED_ORIGINS` is localhost-only. Set the env var to your specific frontend domain in production. Do not use `*`.

**Rate limiting.** Per-IP 60 req/min via `slowapi`. For higher-traffic deployments, put a CDN/WAF (Cloudflare, AWS Shield) in front.

**Dependencies.** `pip-audit` is the canonical check:

```bash
pip install pip-audit && pip-audit
```

Dependabot is enabled.
