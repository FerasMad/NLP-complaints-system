"""
Latency / throughput benchmarks for inference.

Measures:
  - Single-request latency (p50, p95, p99) for warm cache
  - Batch throughput at batch sizes 1, 8, 32
  - Both: single best model AND full ensemble
  - Both: CPU and GPU (if available)

Outputs models/benchmarks.txt with a markdown-friendly table.
"""
from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
ENSEMBLE_CONFIG = ROOT / "models" / "ensemble_final" / "config.json"
LABEL_MAP_JSON = ROOT / "data" / "processed" / "label_map.json"
OUT_TXT = ROOT / "models" / "benchmarks.txt"

with open(LABEL_MAP_JSON, encoding="utf-8") as _f:
    NUM_LABELS = len(json.load(_f))

# 8 prototype texts — varied length, varied dialect, all 8 categories represented
PROTOTYPE_TEXTS = [
    "وصل الطلب بارد جدا والمندوب تاخر اكثر من ساعتين",
    "الاسعار مبالغ فيها لا تناسب الجوده المقدمه ابدا",
    "النظافه سيئه الطاولات متسخه والارض غير نظيفه",
    "طلبت برجر بدون بصل لكنهم وضعوه رغم تنبيهي",
    "الموظف اسلوبه سيء وغير محترم",
    "انتظرت ساعتين قبل ان ياتي الطلب",
    "تجربه سيئه عموما لن اعود",
    "الاكل بايخ ومالح",
]


def load_models(model_dirs, device):
    out = []
    for d in model_dirs:
        tok = AutoTokenizer.from_pretrained(str(d))
        mdl = AutoModelForSequenceClassification.from_pretrained(str(d)).to(device)
        mdl.eval()
        out.append((tok, mdl))
    return out


def predict_one_model(tok, mdl, texts, device, max_length, batch_size):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
            logits = mdl(**enc).logits
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def benchmark_single_request(loaded, texts, device, max_length, n_warmup=5, n_trials=50):
    """Single-text inference latency."""
    # Warmup
    for _ in range(n_warmup):
        for tok, mdl in loaded:
            predict_one_model(tok, mdl, [texts[0]], device, max_length, 1)
    # Trials
    latencies_ms = []
    for trial in range(n_trials):
        text = texts[trial % len(texts)]
        t0 = time.perf_counter()
        probs_stack = np.zeros((len(loaded), NUM_LABELS), dtype=np.float32)
        for k, (tok, mdl) in enumerate(loaded):
            probs_stack[k] = predict_one_model(tok, mdl, [text], device, max_length, 1)[0]
        _ = probs_stack.mean(axis=0).argmax()
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    return latencies_ms


def benchmark_throughput(loaded, texts_pool, device, max_length, batch_size, n_seconds=10):
    """How many requests per second at a given batch size."""
    # Cycle through the pool to simulate continuous load
    n_processed = 0
    t_end = time.perf_counter() + n_seconds
    while time.perf_counter() < t_end:
        # Sample batch_size texts
        idxs = np.random.randint(0, len(texts_pool), size=batch_size)
        batch = [texts_pool[i] for i in idxs]
        for tok, mdl in loaded:
            predict_one_model(tok, mdl, batch, device, max_length, batch_size)
        n_processed += batch_size
    elapsed = n_seconds  # approximate
    return n_processed / elapsed


def run_benchmarks(model_dirs, device_str):
    device = torch.device(device_str)
    print(f"\n=== Loading models on {device_str} ===")
    loaded = load_models(model_dirs, device)
    max_length = 192

    # Latency
    print("Single-request latency (50 trials)...")
    latencies = benchmark_single_request(loaded, PROTOTYPE_TEXTS, device, max_length)
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    mean = statistics.mean(latencies)

    # Throughput
    print("Throughput at batch sizes 1, 8, 32 (10s each)...")
    rng = np.random.RandomState(42)
    pool = PROTOTYPE_TEXTS * 100  # extend the pool
    tps_b1 = benchmark_throughput(loaded, pool, device, max_length, 1)
    tps_b8 = benchmark_throughput(loaded, pool, device, max_length, 8)
    tps_b32 = benchmark_throughput(loaded, pool, device, max_length, 32)

    # Cleanup
    del loaded
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "device": device_str,
        "models": len(model_dirs),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": mean,
        "tps_b1": tps_b1,
        "tps_b8": tps_b8,
        "tps_b32": tps_b32,
    }


def main():
    with open(ENSEMBLE_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    ensemble_dirs = [ROOT / m for m in cfg["models"]]
    single_dir = [ensemble_dirs[1]]  # camelbert-mix_8c_capALL_s2024_final (best single)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    results = []
    for label, dirs in [("single (capALL_s2024)", single_dir), ("ensemble (4 models)", ensemble_dirs)]:
        for device_str in (["cuda", "cpu"] if cuda_available else ["cpu"]):
            print(f"\n========== {label} on {device_str} ==========")
            r = run_benchmarks(dirs, device_str)
            r["label"] = label
            results.append(r)

    lines = ["# Inference benchmarks", ""]
    lines.append("Test environment:")
    if cuda_available:
        lines.append(f"  GPU: {torch.cuda.get_device_name(0)}")
        lines.append(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    lines.append(f"  CUDA available: {cuda_available}")
    lines.append("")
    lines.append("## Single-request latency (lower is better)")
    lines.append("")
    lines.append("| Config | Device | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in results:
        lines.append(f"| {r['label']} | {r['device']} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} | {r['p99_ms']:.1f} | {r['mean_ms']:.1f} |")
    lines.append("")
    lines.append("## Throughput (requests/second; higher is better)")
    lines.append("")
    lines.append("| Config | Device | batch=1 | batch=8 | batch=32 |")
    lines.append("|---|---|---:|---:|---:|")
    for r in results:
        lines.append(f"| {r['label']} | {r['device']} | {r['tps_b1']:.1f} | {r['tps_b8']:.1f} | {r['tps_b32']:.1f} |")

    txt = "\n".join(lines)
    print()
    print(txt)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\nSaved -> {OUT_TXT}")


if __name__ == "__main__":
    main()
