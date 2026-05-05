"""Generate the two charts embedded in the HF Space.

Outputs:
  hf_space/charts/per_class_f1.svg
  hf_space/charts/vs_baselines.svg

Colors match the DESIGN.md tokens (cream surface, terracotta bars, deep ink type).
Run once and commit the SVGs; rerun if numbers change.
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
CREAM = "#F5EFE6"
PAPER = "#EEE6D7"
INK = "#2D211A"
INK_MUTED = "#5C4F45"
TERRACOTTA = "#C75D3D"
TERRACOTTA_DEEP = "#A14828"
OLIVE = "#5F6845"
BORDER = "#D9CFC0"


def _style_axes(ax, *, with_xaxis: bool = False) -> None:
    """Strip axes to a clean editorial look."""
    ax.set_facecolor(CREAM)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=10, length=0)
    ax.tick_params(axis="y", colors=INK, labelsize=11, length=0)
    if not with_xaxis:
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
    ax.grid(False)


def per_class_f1() -> Path:
    """Horizontal bar chart, per-class F1, sorted descending."""
    data = [
        ("جودة الطعام", "Food quality", 0.962),
        ("خدمة الموظفين", "Staff service", 0.958),
        ("النظافة", "Cleanliness", 0.953),
        ("السعر والقيمة", "Price / value", 0.947),
        ("وقت الانتظار", "Wait time", 0.919),
        ("التوصيل", "Delivery", 0.901),
        ("دقة الطلب", "Order accuracy", 0.867),
        ("عامة", "General", 0.849),
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    fig.patch.set_facecolor(CREAM)

    labels = [f"{en}" for _, en, _ in data][::-1]
    values = [v for _, _, v in data][::-1]
    bars = ax.barh(labels, values, color=TERRACOTTA, height=0.62, zorder=3)
    bars[-1].set_color(TERRACOTTA_DEEP)  # the leader

    ax.set_xlim(0.78, 1.0)
    _style_axes(ax)

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%",
            va="center",
            ha="left",
            color=INK,
            fontsize=11,
            fontweight="bold",
        )

    ax.axvline(0.80, color=INK_MUTED, linestyle="--", linewidth=1, alpha=0.45, zorder=1)
    ax.text(
        0.80,
        -0.85,
        "≥ 80% F1 floor",
        color=INK_MUTED,
        fontsize=9,
        ha="center",
        va="top",
        style="italic",
    )

    fig.subplots_adjust(left=0.22, right=0.92, top=0.94, bottom=0.10)
    out = OUT / "per_class_f1.svg"
    fig.savefig(out, format="svg", facecolor=CREAM, transparent=False)
    plt.close(fig)
    return out


def vs_baselines() -> Path:
    """Vertical bar chart of accuracy progression across approaches."""
    data = [
        ("TF-IDF + LinearSVC", 0.898, "9-class baseline"),
        ("CAMeLBERT-mix (single)", 0.932, "9-class single model"),
        ("4-model ensemble", 0.9407, "9-class with ambiance"),
        ("4-model ensemble", 0.9505, "8-class + EDA boost"),
    ]

    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    captions = [d[2] for d in data]

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    fig.patch.set_facecolor(CREAM)

    colors = [PAPER, INK_MUTED, OLIVE, TERRACOTTA]
    bars = ax.bar(range(len(values)), values, color=colors, width=0.55, zorder=3)
    bars[-1].set_color(TERRACOTTA_DEEP)

    ax.set_ylim(0.85, 0.97)
    _style_axes(ax)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=0, fontsize=10, color=INK)

    for i, (bar, val, cap) in enumerate(zip(bars, values, captions)):
        text_color = CREAM if i >= 2 else INK
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
            color=INK_MUTED,
            fontsize=8.5,
            style="italic",
        )

    fig.subplots_adjust(left=0.06, right=0.96, top=0.94, bottom=0.18)
    out = OUT / "vs_baselines.svg"
    fig.savefig(out, format="svg", facecolor=CREAM, transparent=False)
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = per_class_f1()
    p2 = vs_baselines()
    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
