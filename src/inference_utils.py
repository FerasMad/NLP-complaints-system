"""Shared inference utilities — single source of truth for predict_probs.

Replaces the duplicated `predict_probs` functions across eval scripts.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import INFERENCE_BATCH_SIZE


def predict_probs(
    model_dir: str | Path,
    texts: list[str],
    device: torch.device,
    max_length: int = 192,
    num_labels: int | None = None,
) -> np.ndarray:
    """Run a single HuggingFace model on a list of texts, return softmax probs.

    Args:
        model_dir: path or HF Hub id.
        texts: list of cleaned input strings.
        device: torch.device.
        max_length: tokenizer max length.
        num_labels: number of output labels (auto-detected if None).

    Returns:
        Array of shape (len(texts), num_labels) with softmax probabilities.
    """
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    if num_labels is None:
        num_labels = int(model.config.num_labels)

    out = np.zeros((len(texts), num_labels), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), INFERENCE_BATCH_SIZE):
            batch = texts[i : i + INFERENCE_BATCH_SIZE]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(device)
            logits = model(**enc).logits
            out[i : i + len(batch)] = torch.softmax(logits, dim=-1).cpu().numpy()

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def predict_logits(
    model_dir: str | Path,
    texts: list[str],
    device: torch.device,
    max_length: int = 192,
    num_labels: int | None = None,
) -> np.ndarray:
    """Same as predict_probs but returns raw logits (for temperature scaling)."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()
    if num_labels is None:
        num_labels = int(model.config.num_labels)
    out = np.zeros((len(texts), num_labels), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), INFERENCE_BATCH_SIZE):
            batch = texts[i : i + INFERENCE_BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
            logits = model(**enc).logits
            out[i : i + len(batch)] = logits.cpu().numpy()
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
