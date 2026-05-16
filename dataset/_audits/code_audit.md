# Code Audit -- hf_space/app.py
**Audited:** 2026-05-16
**Scope:** Production-readiness of the HuggingFace Space entry point (~2002 lines)
**Diff context:** Visual rail rewrite (lines 443-528) + apply_rescue now returns (out, rescued_idx)
---
## Critical -- Would break production if hit
**None found.** There are no issues in this category.
---
## Important -- Degraded behavior or edge cases### 1. cleaned_text inserted verbatim into HTML -- XSS mitigated today but architecturally fragile
File: hf_space/app.py:335 and hf_space/app.py:371
Verbatim lines:
    # line 335:
    f'  <div class="understanding-text">{cleaned_text}</div>'
    # line 371:
    f'  <div class="understanding-text">{annotated}</div>'
clean() applies NON_ARABIC.sub(" ", t) with pattern r"[^╪Ç-█┐a-zA-Z0-9┘á-┘⌐\s]", which strips every character
that is not an Arabic codepoint, ASCII alphanumeric, or whitespace -- including <, >, double-quote, single-quote,
and &. Classic XSS payloads are neutralized before any HTML is produced. The current call site at line 703
always passes cleaned, so this is safe today.
The fragility: render_understanding() accepts cleaned_text: str with no internal html.escape(). Any future
caller passing raw (un-cleaned) user text -- a test, a new endpoint, a refactor -- would be immediately
injectable.
Fix: html.escape(cleaned_text) at lines 335 and 371, and html.escape(text[s:e]) inside annotate_text() at
line 306 before interpolating into the <mark> tag. This makes the function safe by construction regardless
of how it is called.
---
### 2. Label count mismatch between model output and ID2LABEL would raise unhandled KeyError
File: hf_space/app.py:692 and hf_space/app.py:694
Verbatim lines:
    full_probs = {ID2LABEL[int(i)]: float(probs[i]) for i in range(len(probs))}
    top = [(ID2LABEL[int(i)], float(probs[i])) for i in top_idx]
ID2LABEL is built from the hardcoded 8-entry CATEGORIES list. If the model loaded from HF_REPO_ID has a
different num_labels -- wrong checkpoint deployed, model card updated, or HF_REPO_ID overridden via env var
-- range(len(probs)) exceeds ID2LABEL keys and raises KeyError. predict() has no try/except around the
inference block, so Gradio surfaces a raw traceback to the user instead of a clean error message.
Fix: After line 85 (model load), assert:
    assert model.config.num_labels == len(CATEGORIES), (
        f"Model has {model.config.num_labels} labels but CATEGORIES has {len(CATEGORIES)}"
    )
Optionally, wrap lines 688-695 in try/except Exception that returns
render_message("╪«╪╖╪ú ┘ü┘è ╪º┘ä╪¬╪╡┘å┘è┘ü", "classification error -- please try again").
---
### 3. No raw text length cap before clean() and extract_aspects() -- DoS vector on public Space
File: hf_space/app.py:658
The only length guard is:
    if not text or len(text.strip()) < 3:
There is no upper bound. The tokenizer truncates to MAX_LENGTH=192 tokens, protecting the model. But clean()
(three regex passes) and extract_aspects() (phrase-matching loop over ~50 phrases x full cleaned text length)
run on the complete input. A megabyte of text triggers O(n x vocab_size) character comparisons -- potentially
seconds per request on a public Space with no rate limiting, pegging the CPU and slowing legitimate requests.
Fix: Add text = text[:4000] immediately after the length guard on line 658. Inputs beyond ~1500 Arabic
characters provide no additional model signal given the 192-token truncation.
---
### 4. RESCUE_RULES first-match ordering is implicit policy with no explanation
File: hf_space/app.py:545-603
RESCUE_RULES is ordered with ╪º┘ä┘å╪╕╪º┘ü╪⌐ first and ╪╣╪º┘à╪⌐ last. A text containing both ╪º┘ä╪¡┘à╪º┘à (restroom ->
Cleanliness rescue) and ┘ä┘å ╪º╪╣┘ê╪» (I will not return -> General rescue) will be rescued to Cleanliness; the
General rescue never fires. The break at line 603 documents first-match intent but the ordering rationale
is not stated anywhere. A future maintainer reordering for readability silently changes behavior.
Fix: Add a comment block before RESCUE_RULES stating that order is load-bearing: more specific categories
must come first, ╪╣╪º┘à╪⌐ must be last because its phrase set is intentionally broad and should only fire when
no specific aspect rescue matches.
---
### 5. annotate_text() trusts caller to supply non-overlapping matches -- no internal guard
File: hf_space/app.py:293-312
annotate_text() assumes matches is sorted and non-overlapping. extract_aspects() guarantees this for its own
output via the covered-set dedup at lines 276-283. However, render_understanding() accepts matches as an
optional external argument (line 317), meaning a caller can bypass that guarantee. If matches contains
overlapping spans, out.append(text[last:s]) at line 301 silently emits a zero-length or backward slice,
producing malformed HTML with no error raised.
Fix: Add at the start of the for-loop body (after line 299):
    if s < last:
        continue
This makes annotate_text() safe regardless of caller.
---## Minor -- Cleanup and defensive coding
### 6. apply_rescue probs parameter is untyped; return type changed without annotation update
File: hf_space/app.py:570-575
Verbatim signature:
    def apply_rescue(
        probs,
        cleaned_text: str,
        rescue_floor: float = 0.55,
        other_class_dampening: float = 0.30,
    ):
The return type changed from np.ndarray to tuple[np.ndarray, int | None] in this PR. The signature has no
annotation for either probs or the return type. Fix: add probs: np.ndarray and -> tuple[np.ndarray, int | None].
---
### 7. is_multi_aspect() is dead code after the rail rewrite
File: hf_space/app.py:389-402
Verbatim:
    def is_multi_aspect(
        top: list[tuple[str, float]],
        by_aspect: dict[str, list[str]] | None = None,
    ) -> bool:
        return by_aspect is not None and len(by_aspect) >= 2
No longer called anywhere. render_result() determines multi-aspect directly via len(aspect_cats) >= 2 at
line 517. Remove it, or restore an explicit call so the detection logic is centralized.
---
### 8. Single-word vocab items registered in both Pass 1 and Pass 2 -- redundant work
File: hf_space/app.py:183-187
Single-word entries in ASPECT_VOCAB are added to phrases_by_aspect (Pass 1 surface matching) AND
stem-indexed for Pass 2 when PySarf is available. Both passes can match the same token, producing two raw
entries for the same span. The covered-set dedup at lines 276-283 collapses them correctly, so this is not
a correctness bug -- only redundant iterations and slightly larger raw_matches lists.
---
### 9. print() used for startup logging at module level
File: hf_space/app.py:22, 82, 86
Fine for HuggingFace Spaces (stdout captured by the runtime). Switching to the logging module would allow
level filtering but is not a production blocker.
---
## Looks fine
The NON_ARABIC regex in clean() strips all HTML-special ASCII punctuation from user input before any HTML
construction, so the practical XSS risk from cleaned_text interpolation is very low today. ASPECT_VOCAB,
CATEGORIES, CATEGORIES_EN, and RESCUE_RULES are all static module-level constants; their interpolation into
HTML is safe. All render_message() call sites (lines 659-685) use hardcoded string literals, not user input.
@torch.no_grad() is correctly applied to predict(). model.eval() is correctly set at startup (line 85). The
theme toggle JS wraps localStorage in try/catch, so private-browsing mode does not crash.
The cap=4 logic is correct: aspect_cats[:4] happens after rescued_cat insertion (line 491 before line 494),
so the rescued category is always at index 0 and is never evicted by the cap. The fallback to top[0][0]
when aspect_cats is empty (line 499) correctly handles by_aspect=None, by_aspect={}, and zero-phrase-match
cases identically. The guard list((by_aspect or {}).keys()) at line 478 handles None correctly.
full_probs=None is handled: the sort at lines 481-482 is inside if full_probs is not None and aspect_cats.
rescued_cat=╪╣╪º┘à╪⌐ is guarded at line 488 (if rescued_cat and rescued_cat != ╪╣╪º┘à╪⌐) preventing it from
polluting aspect_cats before the render_general_fallback early-return at line 475. This early-return always
fires when the General rescue is active because the General rescue sets the rescued class to floor 0.55, which
after dampening other classes by 0.30 and renormalization leaves ╪╣╪º┘à╪⌐ as unambiguous top-1.
The apply_rescue s>0 guard at line 612 is sufficient: softmax output is always positive-sum, and the rescue
sets the rescued class to at least rescue_floor=0.55 before renormalization, so s is never zero in the
inference path. demo.launch(server_name=0.0.0.0, server_port=7860) is correct and required for HF Spaces.
---
## Ship Readiness Verdict
Ship it. There are no critical bugs. The one security concern (XSS via cleaned_text interpolation into HTML)
is effectively neutralized by clean() in the current call graph. The two items most worth addressing before
significant traffic growth are the missing raw-text length cap (issue 3 -- one line, prevents CPU-pegging on a
public Space) and the missing startup label-count assertion (issue 2 -- one line, turns a cryptic KeyError
traceback into a clear startup failure if the wrong model checkpoint is deployed). The dead is_multi_aspect()
function (issue 7) and the lack of html.escape() in render_understanding() (issue 1) are the two things most
likely to trip up a future maintainer rather than a current user.