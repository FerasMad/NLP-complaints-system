"""Quick 8-row smoke test against the freshly-deployed Space.

Verifies the new commit's behavior end-to-end:
1. The user-reported headline bug: rail shows ALL detected aspects (3 categories).
2. Single-aspect predictions still work.
3. I-2 (ليس formal MSA): "ليس ممتاز ابدا" should reach the model, not be screened.
4. I-3 (contrastive marker بس): "ممتاز بس الجو حار" should reach the model.
5. Abstain on praise: "ممتاز جدا شكرا" should be screened.
6. Abstain on too-short: "سيء" should be screened.
"""
from __future__ import annotations

import re

from gradio_client import Client

SPACE = "FerasMad/arabic-complaints-classifier"

CASES = [
    ("الاكل جودته سيئه والتوصيل اخذ وقت طويل", "headline 3-aspect rail", "rail-with-3"),
    ("الاكل بايخ ومالح", "single food", "single-food"),
    ("ليس ممتاز ابدا", "I-2 formal MSA negation", "not-praise-screened"),
    ("ممتاز بس الجو حار", "I-3 contrastive marker", "not-praise-screened"),
    ("ممتاز جدا شكرا", "praise -> screened", "praise-screened"),
    ("سيء", "too short", "too-short"),
    ("الحمام كان قذر جدا", "hygiene rescue", "single-hygiene"),
    ("انتظرت ساعه كامله في المطعم", "wait-time rescue", "single-wait"),
]

RAIL_ROW_RE = re.compile(
    r'<div class="result-row\s+result-rank-(top|other)">'
    r'.*?<div class="result-cat">\s*(.*?)\s*</div>',
    re.IGNORECASE | re.DOTALL,
)


def parse(html: str) -> dict:
    if "result-message" in html:
        return {"kind": "abstain", "rail": []}
    rows = RAIL_ROW_RE.findall(html or "")
    cats = [c for _, c in rows]
    return {"kind": "predict", "rail": cats}


def main():
    print(f"Smoke test against {SPACE}\n")
    client = Client(SPACE, verbose=False)
    passed = failed = 0
    for text, label, expect in CASES:
        try:
            html = client.predict(text=text, api_name="/predict")
        except Exception as e:
            print(f"  ERROR [{label}]: {e}")
            failed += 1
            continue
        parsed = parse(html)
        kind, rail = parsed["kind"], parsed["rail"]

        ok = False
        if expect == "rail-with-3":
            ok = kind == "predict" and len(rail) >= 3
            detail = f"kind={kind} rail={rail}"
        elif expect == "single-food":
            ok = kind == "predict" and rail == ["جودة الطعام"]
            detail = f"kind={kind} rail={rail}"
        elif expect == "not-praise-screened":
            ok = kind == "predict"  # must reach the model, not be screened
            detail = f"kind={kind} rail={rail}"
        elif expect == "praise-screened":
            ok = kind == "abstain"
            detail = f"kind={kind}"
        elif expect == "too-short":
            ok = kind == "abstain"
            detail = f"kind={kind}"
        elif expect == "single-hygiene":
            ok = kind == "predict" and "النظافة" in rail
            detail = f"kind={kind} rail={rail}"
        elif expect == "single-wait":
            ok = kind == "predict" and "وقت الانتظار" in rail
            detail = f"kind={kind} rail={rail}"
        else:
            detail = "unknown expectation"

        marker = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {marker} [{label}]")
        print(f"    text: {text!r}")
        print(f"    expect: {expect} | got: {detail}")

    print(f"\nResult: {passed}/{len(CASES)} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
