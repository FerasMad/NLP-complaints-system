"""Reusable ensemble inference module.

Loads N HuggingFace sequence-classification models from a config.json (or env),
predicts by averaging softmax probabilities, applies optional per-class biases,
and supports graceful degradation + OOD abstain.

Used by app/api.py and app/space_app.py.

Public API:
    EnsembleClassifier(config_path) - load + predict
    clean_arabic(text) - Lana's text cleaning
    is_arabic_enough(text) - language gate (used for OOD abstain)
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# --- Lana's text cleaning ---
TASHKEEL = re.compile(r"[ً-ٰٟؐ-ؚ]")
NON_ARABIC = re.compile(r"[^؀-ۿa-zA-Z0-9٠-٩\s]")
WHITESPACE = re.compile(r"\s+")
ARABIC_CHAR = re.compile(r"[؀-ۿ]")


def clean_arabic(text: str) -> str:
    """Apply Lana's text cleaning pipeline.

    Steps: remove tashkeel, normalize alef/ya/ta-marbuta, strip non-Arabic
    punctuation, collapse whitespace, lowercase ASCII.

    Args:
        text: raw input string (may contain dialect, English, emojis).

    Returns:
        Cleaned, normalized Arabic-friendly string. Empty if input is empty.
    """
    if not text:
        return ""
    t = TASHKEEL.sub("", text)
    t = t.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"}))
    t = NON_ARABIC.sub(" ", t)
    return WHITESPACE.sub(" ", t).strip().lower()


def arabic_ratio(text: str) -> float:
    """Fraction of characters that are Arabic-script.

    Returns 0.0 for empty / very short input.
    """
    if not text or len(text) < 3:
        return 0.0
    arabic_chars = len(ARABIC_CHAR.findall(text))
    return arabic_chars / max(len(text), 1)


def is_arabic_enough(text: str, min_ratio: float = 0.30) -> bool:
    """Gate for OOD detection: True iff text has enough Arabic-script chars."""
    return arabic_ratio(text) >= min_ratio


# --- Keyword priors: defensive layer for known model-failure patterns ---
# The trained ensemble over-predicts "جودة الطعام" (47% of training data) as a
# fallback when uncertain. These keyword sets are evidence the model sometimes
# misses, especially for categories under-represented in training. Patterns are
# checked against the cleaned (normalized) input.

KEYWORD_PRIORS: dict[str, list[str]] = {
    "النظافة": [
        "الحمام", "حمام", "تواليت", "متسخ", "متسخه", "وسخ",
        "اوساخ", "ذباب", "صراصير", "قذر", "قذره", "نظاف",
    ],
    "وقت الانتظار": [
        "انتظرت", "انتظرنا", "ساعه كامله", "ساعتين كامل",
        "وقت طويل", "تاخير في المطعم", "جلسنا ساعه",
        "قبل ان ياتي", "ما حد جا", "نص ساعه", "نصف ساعه",
        "نصف ساعه", "بطء الخدمه",
    ],
    "التوصيل": [
        "المندوب", "السائق", "ضاع الطلب", "العنوان غلط",
        "ديليفري", "موصل الطلب", "تاخر التوصيل",
    ],
    "خدمة الموظفين": [
        "موظف وقح", "النادل", "الكاشير", "غير محترم",
        "صاح علي", "اسلوبه سيء", "ما يبتسم", "موظف غير",
    ],
    "السعر والقيمة": [
        "غاليه", "غالي جدا", "مبالغ", "ما يستاهل",
        "الفاتوره", "اسعارهم", "ما يستحق",
    ],
    "دقة الطلب": [
        "ناقص", "نسوا", "غلط في الطلب", "الطلب غلط", "الطلب خطا",
        "بدلوا الطلب", "وضعوا بدلا", "خلطوا الطلب", "جاني غلط",
    ],
}

# Strong rescue: phrases so unambiguous that they must produce the matching
# category as top-1, even if the model is highly confident on something else.
RESCUE_RULES: list[tuple[str, list[str]]] = [
    ("النظافة", ["الحمام", "تواليت", "ذباب", "صراصير"]),
    ("وقت الانتظار", [
        "انتظرت ساعه", "انتظرت ساعتين", "ساعه كامله في المطعم",
        "انتظرنا ساعه", "انتظرنا ساعتين",
    ]),
    ("التوصيل", ["ضاع الطلب", "المندوب تاخر", "المندوب ما رد"]),
    ("جودة الطعام", ["الطبخ", "اللحم محروق", "بدون طعم", "الاكل بايخ"]),
]

# Food-quality is the over-predicted fallback. If no food-related phrase appears
# in the input but the model gives food-quality > 0.5, gently penalize so other
# evidence has room to dominate.
FOOD_QUALITY_KEYWORDS: list[str] = [
    "الاكل", "الطعام", "البرجر", "الطبخ", "طبخ", "اللحم", "الدجاج",
    "بايخ", "مالح", "محروق", "نيء", "طعم", "مذاق", "بارد",
    "بدون طعم", "نكهه", "متقن", "الوجبه",
]


KEYWORD_PRIORS["جودة الطعام"] = [
    "الطبخ", "طبخ", "متقن", "الوجبه", "اللحم محروق", "بدون طعم",
    "بايخ", "مالح",
]


def _keyword_match(text: str, patterns: list[str]) -> bool:
    return any(p in text for p in patterns)


def apply_keyword_priors(
    probs: np.ndarray,
    cleaned_text: str,
    label2id: dict,
    *,
    rescue_floor: float = 0.55,
) -> np.ndarray:
    """Apply defensive keyword evidence to model probabilities.

    Conservative single-layer design (after audit showed soft-boost variants
    regress on held-out test): only the strong rescue activates, and only
    on unambiguous phrases. This catches the most egregious failures (e.g.
    bathroom complaints classified as food quality) without over-firing.

    All operations preserve the simplex (sum = 1).
    """
    out = probs.copy()

    for cat, phrases in RESCUE_RULES:
        if cat not in label2id:
            continue
        if _keyword_match(cleaned_text, phrases):
            cat_idx = label2id[cat]
            current = float(out.max())
            out[cat_idx] = max(out[cat_idx], current + 0.05, rescue_floor)

    s = out.sum()
    if s > 0:
        out = out / s
    return out


def softmax_entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a probability vector (natural log)."""
    p = np.clip(probs, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


@dataclass
class PredictionResult:
    """Structured prediction output.

    Attributes:
        category: top-1 category name, OR None when abstaining.
        confidence: top-1 probability (post-bias).
        top_3: list of (category, confidence) sorted descending.
        abstain_reason: None on normal prediction; otherwise one of
            "out_of_domain" (text is not Arabic), "low_confidence" (top
            prob below threshold), "high_entropy" (uncertain), "too_short".
        request_meta: optional debug dict (latency_ms, models_used, etc.).
    """
    category: Optional[str]
    confidence: float
    top_3: list[tuple[str, float]]
    abstain_reason: Optional[str] = None
    request_meta: Optional[dict] = None


class EnsembleClassifier:
    """Average-softmax ensemble of N HuggingFace sequence classifiers.

    Features:
    - Graceful degradation: if one model fails to load, continue with the
      rest (raises only if fewer than 2 models load).
    - OOD abstain: returns None category if input has too few Arabic chars
      or model entropy is too high.
    - Optional per-class additive biases (from config).
    - Model revision pinning via per-model "revision" field in config.

    Usage:
        clf = EnsembleClassifier("models/ensemble_final/config.json")
        result = clf.predict("الاكل بايخ")
        print(result.category, result.confidence)
    """

    def __init__(
        self,
        config_path: Path | str,
        project_root: Optional[Path | str] = None,
        min_arabic_ratio: float = 0.30,
        entropy_abstain_threshold: Optional[float] = None,
        confidence_abstain_threshold: float = 0.0,
    ):
        """Load the ensemble.

        Args:
            config_path: path to ensemble config.json (manifest of model dirs + biases).
            project_root: project root for resolving relative model paths.
                Defaults to the parent of app/.
            min_arabic_ratio: minimum fraction of Arabic-script chars to accept
                input (else abstain with reason="out_of_domain").
            entropy_abstain_threshold: if set, abstain when softmax entropy >
                this value. None disables.
            confidence_abstain_threshold: if top-1 confidence is below this,
                abstain. 0.0 disables.

        Raises:
            RuntimeError: if fewer than 2 models load successfully (degraded
                ensemble below this point is unsafe).
        """
        config_path = Path(config_path)
        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)

        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent
        self.project_root = Path(project_root)
        self.max_length = int(self.config.get("max_length", 192))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.min_arabic_ratio = min_arabic_ratio
        self.entropy_abstain_threshold = entropy_abstain_threshold
        self.confidence_abstain_threshold = confidence_abstain_threshold

        # Load label map
        with open(self.project_root / self.config["label_map_path"], encoding="utf-8") as f:
            label_map = json.load(f)
        self.id2label: dict[int, str] = {int(idx): cat for cat, idx in label_map.items()}
        self.label2id: dict[str, int] = {cat: i for i, cat in self.id2label.items()}
        self.num_labels = len(self.id2label)

        # Load all models with graceful degradation
        self.models: list = []
        self.tokenizers: list = []
        self.loaded_model_names: list[str] = []
        load_errors: list[str] = []

        # Optional per-model revisions for pinning
        revisions: dict[str, str] = self.config.get("model_revisions", {})

        t_start = time.perf_counter()
        for spec in self.config["models"]:
            # spec can be a string path/HF id or a dict {path, revision}
            if isinstance(spec, dict):
                path_or_id = spec.get("path") or spec.get("id") or ""
                revision = spec.get("revision") or revisions.get(path_or_id)
            else:
                path_or_id = spec
                revision = revisions.get(path_or_id)

            full_path: str
            if (self.project_root / path_or_id).exists():
                full_path = str(self.project_root / path_or_id)
            else:
                # Treat as a HuggingFace Hub model id
                full_path = path_or_id

            try:
                logger.info(f"  loading {full_path}{' @ ' + revision if revision else ''} ...")
                tok_kwargs = {"revision": revision} if revision else {}
                mdl_kwargs = {"revision": revision} if revision else {}
                tok = AutoTokenizer.from_pretrained(full_path, **tok_kwargs)
                mdl = AutoModelForSequenceClassification.from_pretrained(full_path, **mdl_kwargs).to(self.device)
                mdl.eval()
                self.tokenizers.append(tok)
                self.models.append(mdl)
                self.loaded_model_names.append(path_or_id)
            except Exception as exc:
                msg = f"failed to load {full_path}: {type(exc).__name__}: {exc}"
                load_errors.append(msg)
                logger.warning(msg)

        if len(self.models) < 2:
            raise RuntimeError(
                f"Ensemble degraded below safe minimum: only {len(self.models)} model(s) "
                f"loaded out of {len(self.config['models'])}. Errors: {load_errors}"
            )
        if load_errors:
            warnings.warn(
                f"Ensemble degraded: {len(self.models)}/{len(self.config['models'])} models loaded. "
                f"Errors: {load_errors}",
                stacklevel=2,
            )

        # Per-class additive biases
        biases = self.config.get("tuned_biases", {})
        self.biases = np.zeros(self.num_labels, dtype=np.float32)
        for i, cat in self.id2label.items():
            self.biases[i] = float(biases.get(cat, 0.0))

        self.load_seconds = time.perf_counter() - t_start
        logger.info(
            f"Ensemble loaded: {len(self.models)} models on {self.device} "
            f"(took {self.load_seconds:.1f}s)"
        )

    @torch.no_grad()
    def _predict_probs_raw(self, text: str) -> np.ndarray:
        """Run all models, return averaged softmax probabilities (num_labels,).

        Returns the post-cleaning, pre-bias probabilities. For deterministic
        output, the cleaned text is the same on every call.
        """
        cleaned = clean_arabic(text)
        if len(cleaned) < 3:
            return np.zeros(self.num_labels, dtype=np.float32)

        probs_stack = np.zeros((len(self.models), self.num_labels), dtype=np.float32)
        for k, (tok, mdl) in enumerate(zip(self.tokenizers, self.models)):
            enc = tok(cleaned, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
            logits = mdl(**enc).logits[0]
            probs_stack[k] = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs_stack.mean(axis=0)

    def predict_probs(self, text: str) -> np.ndarray:
        """Public alias of _predict_probs_raw (used by tests)."""
        return self._predict_probs_raw(text)

    def predict(
        self,
        text: str,
        top_k: int = 3,
        include_meta: bool = False,
    ) -> PredictionResult:
        """Classify a single text into one of the categories.

        Args:
            text: raw input. Will be cleaned automatically.
            top_k: number of top-k predictions to include in the response.
            include_meta: if True, include latency_ms + ensemble_size in result.

        Returns:
            PredictionResult. If the input is too short, not Arabic enough, or
            the prediction entropy is too high, `category` will be None and
            `abstain_reason` will explain why.
        """
        t_start = time.perf_counter() if include_meta else None

        # OOD gate: text too short
        if not text or len(text.strip()) < 3:
            return PredictionResult(
                category=None,
                confidence=0.0,
                top_3=[],
                abstain_reason="too_short",
                request_meta=None,
            )

        # OOD gate: not enough Arabic
        if not is_arabic_enough(text, min_ratio=self.min_arabic_ratio):
            return PredictionResult(
                category=None,
                confidence=0.0,
                top_3=[],
                abstain_reason="out_of_domain",
                request_meta=None,
            )

        probs = self._predict_probs_raw(text)
        # Defensive keyword post-processing
        cleaned = clean_arabic(text)
        probs = apply_keyword_priors(probs, cleaned, self.label2id)
        adjusted = probs + self.biases

        # Entropy gate (uncertainty abstain)
        entropy = softmax_entropy(probs)
        if (
            self.entropy_abstain_threshold is not None
            and entropy > self.entropy_abstain_threshold
        ):
            idx_sorted = adjusted.argsort()[::-1]
            top_pairs = [(self.id2label[int(i)], float(probs[i])) for i in idx_sorted[:top_k]]
            return PredictionResult(
                category=None,
                confidence=float(probs[idx_sorted[0]]),
                top_3=top_pairs,
                abstain_reason="high_entropy",
                request_meta=None,
            )

        idx_sorted = adjusted.argsort()[::-1]
        pairs = [(self.id2label[int(i)], float(probs[i])) for i in idx_sorted[:top_k]]
        top_cat, top_conf = pairs[0]

        # Confidence gate
        if top_conf < self.confidence_abstain_threshold:
            return PredictionResult(
                category=None,
                confidence=top_conf,
                top_3=pairs,
                abstain_reason="low_confidence",
                request_meta=None,
            )

        meta = None
        if include_meta:
            meta = {
                "latency_ms": (time.perf_counter() - t_start) * 1000,
                "ensemble_size": len(self.models),
                "loaded_models": list(self.loaded_model_names),
                "entropy": entropy,
                "device": str(self.device),
            }

        return PredictionResult(
            category=top_cat,
            confidence=top_conf,
            top_3=pairs,
            abstain_reason=None,
            request_meta=meta,
        )


# Quick smoke test when run directly
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = Path(__file__).resolve().parent.parent / "models" / "ensemble_final" / "config.json"
    clf = EnsembleClassifier(cfg)

    samples = [
        ("الاكل بايخ ومالح", "food"),
        ("الاسعار مبالغ فيها", "price"),
        ("Hello, how are you?", "abstain (English)"),
        ("aaaa", "abstain (too short / no Arabic)"),
        ("ساعتين انتظار", "wait"),
    ]
    for s, expected in samples:
        r = clf.predict(s, include_meta=True)
        cat = r.category or f"ABSTAIN ({r.abstain_reason})"
        meta = f"  [{r.request_meta['latency_ms']:.1f}ms]" if r.request_meta else ""
        print(f"[{expected:<30s}]  {s[:40]:<40s} -> {cat} ({r.confidence:.0%}){meta}")
