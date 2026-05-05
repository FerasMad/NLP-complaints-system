"""Generate the four charts embedded in the HF Space (light + dark themes).

Outputs:
  hf_space/charts/per_class_f1.svg
  hf_space/charts/per_class_f1.dark.svg
  hf_space/charts/vs_baselines.svg
  hf_space/charts/vs_baselines.dark.svg

Colors match the DESIGN.md tokens. Run once and commit; rerun if numbers change.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "hf_space" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# Design tokens (must match hf_space/app.py CSS variables)
LIGHT = {
    "surface": "#F5EFE6",
    "paper": "#EEE6D7",
    "ink": "#2D211A",
    "ink_muted": "#5C4F45",
    "terracotta": "#C75D3D",
    "terracotta_deep": "#A14828",
    "olive": "#5F6845",
    "border": "#D9CFC0",
}

DARK = {
    "surface": "#1A140F",
    "paper": "#241B14",
    "ink": "#F0E7D8",
    "ink_muted": "#A89C8C",
    "terracotta": "#D87852",
    "terracotta_deep": "#C75D3D",
    "olive": "#8A9468",
    "border": "#3A2D24",
}

PER_CLASS_DATA = [
    ("جودة الطعام", "Food quality", 0.962),
    ("خدمة الموظفين", "Staff service", 0.958),
    ("النظافة", "Cleanliness", 0.953),
    ("السعر والقيمة", "Price / value", 0.947),
    ("وقت الانتظار", "Wait time", 0.919),
    ("التوصيل", "Delivery", 0.901),
    ("دقة الطلب", "Order accuracy", 0.867),
    ("عامة", "General", 0.849),
]

BASELINES_DATA = [
    ("TF-IDF + LinearSVC", 0.898, "9-class baseline"),
    ("CAMeLBERT-mix (single)", 0.932, "9-class single model"),
    ("4-model ensemble", 0.9407, "9-class with ambiance"),
    ("4-model ensemble", 0.9505, "8-class + EDA boost"),
]


def _style_axes(ax, palette, *, with_xaxis: bool = False) -> None:
    ax.set_facecolor(palette["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(palette["border"])
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="x", colors=palette["ink_muted"], labelsize=10, length=0)
    ax.tick_params(axis="y", colors=palette["ink"], labelsize=11, length=0)
    if not with_xaxis:
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
    ax.grid(False)


def per_class_f1(palette, suffix: str) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    fig.patch.set_facecolor(palette["surface"])

    labels = [en for _, en, _ in PER_CLASS_DATA][::-1]
    values = [v for _, _, v in PER_CLASS_DATA][::-1]
    bars = ax.barh(labels, values, color=palette["terracotta"], height=0.62, zorder=3)
    bars[-1].set_color(palette["terracotta_deep"])

    ax.set_xlim(0.78, 1.0)
    _style_axes(ax, palette)

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%",
            va="center",
            ha="left",
            color=palette["ink"],
            fontsize=11,
            fontweight="bold",
        )

    ax.axvline(0.80, color=palette["ink_muted"], linestyle="--", linewidth=1, alpha=0.45, zorder=1)
    ax.text(
        0.80,
        -0.85,
        "≥ 80% F1 floor",
        color=palette["ink_muted"],
        fontsize=9,
        ha="center",
        va="top",
        style="italic",
    )

    fig.subplots_adjust(left=0.22, right=0.92, top=0.94, bottom=0.10)
    out = OUT / f"per_class_f1{suffix}.svg"
    fig.savefig(out, format="svg", facecolor=palette["surface"], transparent=False)
    plt.close(fig)
    return out


def vs_baselines(palette, suffix: str) -> Path:
    labels = [d[0] for d in BASELINES_DATA]
    values = [d[1] for d in BASELINES_DATA]
    captions = [d[2] for d in BASELINES_DATA]

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    fig.patch.set_facecolor(palette["surface"])

    colors = [palette["paper"], palette["ink_muted"], palette["olive"], palette["terracotta"]]
    bars = ax.bar(range(len(values)), values, color=colors, width=0.55, zorder=3)
    bars[-1].set_color(palette["terracotta_deep"])

    ax.set_ylim(0.85, 0.97)
    _style_axes(ax, palette)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=0, fontsize=10, color=palette["ink"])

    for i, (bar, val, cap) in enumerate(zip(bars, values, captions)):
        text_color = palette["surface"] if i >= 2 else palette["ink"]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 0.005,
            f"{val * 100:.1f}%",
            ha="center",
            va="top",
            color=text_color,
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.852,
            cap,
            ha="center",
            va="bottom",
            color=palette["ink_muted"],
            fontsize=8.5,
            style="italic",
        )

    fig.subplots_adjust(left=0.06, right=0.96, top=0.94, bottom=0.18)
    out = OUT / f"vs_baselines{suffix}.svg"
    fig.savefig(out, format="svg", facecolor=palette["surface"], transparent=False)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for palette, suffix in [(LIGHT, ""), (DARK, ".dark")]:
        p1 = per_class_f1(palette, suffix)
        p2 = vs_baselines(palette, suffix)
        print(f"Wrote {p1}")
        print(f"Wrote {p2}")
