#!/usr/bin/env python3
"""
Generate professional benchmark figures from results.csv.

Creates 6 publication-quality figures that differentiate quantum and classical
models using consistent colour palettes, hatching, and annotations.

Output: benchmarks/figures/*.png  (300 dpi, tight layout)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# ── paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "results.csv")
FIG_DIR    = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── load data ──
df = pd.read_csv(CSV_PATH)
# Clean param column (may contain strings like "~916,992 nodes")
df["n_params_num"] = pd.to_numeric(
    df["n_params"].astype(str).str.replace(r"[^\d.]", "", regex=True),
    errors="coerce",
).fillna(0).astype(int)
df = df.sort_values("latent_MSE").reset_index(drop=True)

# ── classify quantum vs classical ──
QUANTUM_NAMES = {
    "Simple PML + Ridge",
    "★ QORC + Ridge (ours)",
    "QUANTECH MLP (ours)",
    "Quantum LSTM",
    "VQC (Trained)",
}
df["is_quantum"] = df["name"].isin(QUANTUM_NAMES)

# Short display names
SHORT = {
    "Simple PML + Ridge":     "Simple PML\n+Ridge",
    "Ridge Regression":       "Ridge",
    "sklearn MLP":            "sklearn MLP",
    "★ QORC + Ridge (ours)":  "★ QORC+Ridge\n(ours)",
    "Classical LSTM":         "LSTM",
    "Random Forest":          "Random\nForest",
    "QUANTECH MLP (ours)":    "QUANTECH\nMLP",
    "SVR (RBF)":              "SVR",
    "Gradient Boosting":      "Gradient\nBoosting",
    "Quantum LSTM":           "Quantum\nLSTM",
    "VQC (Trained)":          "VQC",
}
df["short"] = df["name"].map(SHORT)

# ── colour & style palettes ──
QUANTUM_COLOR   = "#8b5cf6"   # vivid purple
CLASSICAL_COLOR = "#10b981"   # emerald green
OUR_MODEL_COLOR = "#c084fc"   # lighter purple (star highlight)

BG_DARK   = "#0f0f17"
BG_CARD   = "#16161f"
GRID_CLR  = "#2a2a3a"
TEXT_CLR   = "#e5e7eb"
MUTED_CLR  = "#9ca3af"

def bar_color(row):
    if row["name"] == "★ QORC + Ridge (ours)":
        return OUR_MODEL_COLOR
    return QUANTUM_COLOR if row["is_quantum"] else CLASSICAL_COLOR

def edge_color(row):
    if row["name"] == "★ QORC + Ridge (ours)":
        return "#a855f7"
    return "#7c3aed" if row["is_quantum"] else "#059669"

df["color"]      = df.apply(bar_color, axis=1)
df["edgecolor"]  = df.apply(edge_color, axis=1)
df["hatch"]      = df["is_quantum"].map({True: "//", False: ""})

# ── global matplotlib style ──
plt.rcParams.update({
    "figure.facecolor":   BG_DARK,
    "axes.facecolor":     BG_CARD,
    "axes.edgecolor":     GRID_CLR,
    "axes.labelcolor":    TEXT_CLR,
    "axes.grid":          True,
    "grid.color":         GRID_CLR,
    "grid.linewidth":     0.4,
    "xtick.color":        MUTED_CLR,
    "ytick.color":        MUTED_CLR,
    "text.color":         TEXT_CLR,
    "font.family":        "sans-serif",
    "font.size":          11,
    "legend.facecolor":   BG_CARD,
    "legend.edgecolor":   GRID_CLR,
    "legend.labelcolor":  TEXT_CLR,
    "savefig.facecolor":  BG_DARK,
    "savefig.dpi":        300,
})

LEGEND_HANDLES = [
    Patch(facecolor=QUANTUM_COLOR, edgecolor="#7c3aed", label="Quantum", hatch="//"),
    Patch(facecolor=CLASSICAL_COLOR, edgecolor="#059669", label="Classical"),
    Patch(facecolor=OUR_MODEL_COLOR, edgecolor="#a855f7", label="★ Ours (QORC+Ridge)", hatch="//"),
]


def add_legend(ax, loc="upper right"):
    ax.legend(handles=LEGEND_HANDLES, loc=loc, fontsize=9,
              framealpha=0.85, borderpad=0.6)


def annotate_bars(ax, bars, values, fmt="{:.4f}", offset=4):
    """Put value labels on top of bars."""
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + offset * 0.001,
            fmt.format(val), ha="center", va="bottom",
            fontsize=7.5, color=MUTED_CLR, fontweight="medium",
        )


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════
#  FIGURE 1 — Latent MSE bar chart
# ═══════════════════════════════════════════════════════════
def fig_latent_mse():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(df))
    bars = ax.bar(
        x, df["latent_MSE"], width=0.7,
        color=df["color"], edgecolor=df["edgecolor"], linewidth=1.2,
        hatch=df["hatch"].tolist(), zorder=3,
    )
    annotate_bars(ax, bars, df["latent_MSE"], fmt="{:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(df["short"], fontsize=9)
    ax.set_ylabel("Latent MSE  (lower is better)", fontsize=11, fontweight="medium")
    ax.set_title("Latent-Space MSE — All 11 Models", fontsize=15, fontweight="bold", pad=14)
    ax.set_ylim(0, df["latent_MSE"].max() * 1.18)
    # Highlight our model
    our_idx = df.index[df["name"] == "★ QORC + Ridge (ours)"][0]
    ax.annotate(
        "★ Our model", xy=(our_idx, df.loc[our_idx, "latent_MSE"]),
        xytext=(our_idx + 1.6, df.loc[our_idx, "latent_MSE"] + 0.012),
        fontsize=9, fontweight="bold", color=OUR_MODEL_COLOR,
        arrowprops=dict(arrowstyle="->", color=OUR_MODEL_COLOR, lw=1.5),
    )
    add_legend(ax)
    save(fig, "01_latent_mse.png")

# ═══════════════════════════════════════════════════════════
#  FIGURE 2 — R² Score bar chart
# ═══════════════════════════════════════════════════════════
def fig_r2():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(df))
    bars = ax.bar(
        x, df["R2"], width=0.7,
        color=df["color"], edgecolor=df["edgecolor"], linewidth=1.2,
        hatch=df["hatch"].tolist(), zorder=3,
    )
    # Value labels
    for bar, val in zip(bars, df["R2"]):
        y = max(bar.get_height(), 0) + 0.015
        ax.text(
            bar.get_x() + bar.get_width() / 2, y,
            f"{val:.3f}", ha="center", va="bottom",
            fontsize=7.5, color=MUTED_CLR, fontweight="medium",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(df["short"], fontsize=9)
    ax.set_ylabel("R² Score  (higher is better)", fontsize=11, fontweight="medium")
    ax.set_title("R² Score — All 11 Models", fontsize=15, fontweight="bold", pad=14)
    ax.axhline(0, color="#ef4444", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
    ax.set_ylim(min(df["R2"].min() - 0.08, -0.3), 1.0)
    add_legend(ax)
    save(fig, "02_r2_score.png")

# ═══════════════════════════════════════════════════════════
#  FIGURE 3 — Surface RMSE bar chart
# ═══════════════════════════════════════════════════════════
def fig_surface_rmse():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(df))
    bars = ax.bar(
        x, df["surface_RMSE"], width=0.7,
        color=df["color"], edgecolor=df["edgecolor"], linewidth=1.2,
        hatch=df["hatch"].tolist(), zorder=3,
    )
    annotate_bars(ax, bars, df["surface_RMSE"], fmt="{:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(df["short"], fontsize=9)
    ax.set_ylabel("Surface RMSE  (lower is better)", fontsize=11, fontweight="medium")
    ax.set_title("Surface RMSE (Price-Space Error) — All 11 Models", fontsize=15, fontweight="bold", pad=14)
    ax.set_ylim(0, df["surface_RMSE"].max() * 1.18)
    add_legend(ax)
    save(fig, "03_surface_rmse.png")

# ═══════════════════════════════════════════════════════════
#  FIGURE 4 — Accuracy vs Inference Speed (scatter)
# ═══════════════════════════════════════════════════════════
def fig_accuracy_vs_speed():
    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in df.iterrows():
        is_ours = row["name"] == "★ QORC + Ridge (ours)"
        marker = "*" if is_ours else ("D" if row["is_quantum"] else "o")
        size = 220 if is_ours else 110
        zorder = 10 if is_ours else 5
        ax.scatter(
            row["inference_ms"], row["latent_MSE"],
            c=row["color"], edgecolors=row["edgecolor"],
            s=size, marker=marker, linewidths=1.5, zorder=zorder,
        )
        # label
        nudge_x, nudge_y = 1.12, 1.0
        ha = "left"
        name = row["short"].replace("\n", " ")
        if row["inference_ms"] > 100:
            ha = "right"
            nudge_x = 0.88
        ax.annotate(
            name,
            xy=(row["inference_ms"], row["latent_MSE"]),
            xytext=(row["inference_ms"] * nudge_x, row["latent_MSE"] * nudge_y + 0.001),
            fontsize=7.5, color=MUTED_CLR, ha=ha, va="bottom",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference Latency (ms, log scale)  →  slower", fontsize=11, fontweight="medium")
    ax.set_ylabel("← better    Latent MSE    worse →", fontsize=11, fontweight="medium")
    ax.set_title("Accuracy vs Inference Speed", fontsize=15, fontweight="bold", pad=14)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.2f}ms" if v < 1 else (f"{v:.0f}ms" if v < 1000 else f"{v/1000:.1f}s")
    ))

    # Ideal zone annotation
    ax.annotate(
        "  Ideal zone\n  (low MSE, fast)",
        xy=(0.04, 0.008), fontsize=9, fontstyle="italic",
        color="#6ee7b7", alpha=0.7,
    )
    add_legend(ax, loc="upper left")
    save(fig, "04_accuracy_vs_speed.png")

# ═══════════════════════════════════════════════════════════
#  FIGURE 5 — Training Time (log bar)
# ═══════════════════════════════════════════════════════════
def fig_training_time():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(df))
    times = df["train_time"].replace(0, 0.001)  # avoid log(0)
    bars = ax.bar(
        x, times, width=0.7,
        color=df["color"], edgecolor=df["edgecolor"], linewidth=1.2,
        hatch=df["hatch"].tolist(), zorder=3,
    )
    # Value labels
    for bar, val in zip(bars, df["train_time"]):
        label = f"{val:.2f}s" if val < 10 else f"{val:.1f}s"
        if val == 0:
            label = "pre-comp."
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            label, ha="center", va="bottom",
            fontsize=7.5, color=MUTED_CLR, fontweight="medium",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(df["short"], fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("Training Time (seconds, log scale)", fontsize=11, fontweight="medium")
    ax.set_title("Training Time — All 11 Models", fontsize=15, fontweight="bold", pad=14)
    add_legend(ax)
    save(fig, "05_training_time.png")

# ═══════════════════════════════════════════════════════════
#  FIGURE 6 — Grouped Quantum vs Classical summary
# ═══════════════════════════════════════════════════════════
def fig_quantum_vs_classical():
    """Side-by-side grouped comparison of quantum vs classical averages."""
    q = df[df["is_quantum"]]
    c = df[~df["is_quantum"]]

    # Exclude VQC & QLSTM from quantum avg (they're negative examples)
    q_good = q[~q["name"].isin(["VQC (Trained)", "Quantum LSTM"])]

    metrics = {
        "Latent MSE ↓":    ("latent_MSE",   False),
        "Surface RMSE ↓":  ("surface_RMSE", False),
        "R² Score ↑":      ("R2",           True),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        "Quantum (top 3) vs Classical (all 6) — Average Metrics",
        fontsize=15, fontweight="bold", y=1.02,
    )

    for ax, (label, (col, higher_better)) in zip(axes, metrics.items()):
        q_val = q_good[col].mean()
        c_val = c[col].mean()

        bars = ax.bar(
            [0, 1], [q_val, c_val], width=0.55,
            color=[QUANTUM_COLOR, CLASSICAL_COLOR],
            edgecolor=["#7c3aed", "#059669"],
            linewidth=1.5, hatch=["//", ""], zorder=3,
        )
        # Value labels
        for bar, val in zip(bars, [q_val, c_val]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(bar.get_height()) * 0.03 + 0.001,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=11, color=TEXT_CLR, fontweight="bold",
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Quantum\n(top 3)", "Classical\n(all 6)"], fontsize=10, fontweight="medium")
        ax.set_title(label, fontsize=12, fontweight="bold", pad=8)

        ymin = min(q_val, c_val)
        ymax = max(q_val, c_val)
        margin = (ymax - ymin) * 0.4 + 0.002
        ax.set_ylim(max(0, ymin - margin), ymax + margin)

        # Winner arrow
        if (higher_better and q_val > c_val) or (not higher_better and q_val < c_val):
            winner_label = "Quantum wins"
            winner_color = QUANTUM_COLOR
        else:
            winner_label = "Classical wins"
            winner_color = CLASSICAL_COLOR
        ax.text(0.5, 0.02, winner_label, ha="center", va="bottom",
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                color=winner_color, alpha=0.85)

    fig.tight_layout()
    save(fig, "06_quantum_vs_classical.png")


# ═══════════════════════════════════════════════════════════
#  Run all figures
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating benchmark figures...")
    print(f"  Source: {CSV_PATH}")
    print(f"  Output: {FIG_DIR}/")
    print()
    fig_latent_mse()
    fig_r2()
    fig_surface_rmse()
    fig_accuracy_vs_speed()
    fig_training_time()
    fig_quantum_vs_classical()
    print()
    print(f"Done — 6 figures saved to {FIG_DIR}/")
