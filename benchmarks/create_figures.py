#!/usr/bin/env python3
"""
Generate professional benchmark figures (light theme, presentation-ready).

Inspired by the EPFL-Quandela Qedi team's website aesthetic:
  - White / light backgrounds for web embedding
  - Clean Inter-like sans-serif typography
  - Purple accent for quantum, emerald for classical
  - Rounded corners, subtle grid, publication quality

Output: benchmarks/figures/*.png  (300 dpi, tight layout)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.patches import Patch, FancyBboxPatch

# ── paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "results.csv")
FIG_DIR    = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── load data ──
df = pd.read_csv(CSV_PATH)
df["n_params_num"] = pd.to_numeric(
    df["n_params"].astype(str).str.replace(r"[^\d.]", "", regex=True),
    errors="coerce",
).fillna(0).astype(int)
df = df.sort_values("latent_MSE").reset_index(drop=True)

# ── classify ──
QUANTUM_NAMES = {
    "Simple PML + Ridge",
    "★ QORC + Ridge (ours)",
    "QUANTECH MLP (ours)",
    "Quantum LSTM",
    "VQC (Trained)",
}
df["is_quantum"] = df["name"].isin(QUANTUM_NAMES)
df["is_ours"]    = df["name"] == "★ QORC + Ridge (ours)"

SHORT = {
    "Simple PML + Ridge":     "Simple PML+Ridge",
    "Ridge Regression":       "Ridge",
    "sklearn MLP":            "sklearn MLP",
    "★ QORC + Ridge (ours)":  "★ QORC+Ridge",
    "Classical LSTM":         "Classical LSTM",
    "Random Forest":          "Random Forest",
    "QUANTECH MLP (ours)":    "QUANTECH MLP",
    "SVR (RBF)":              "SVR (RBF)",
    "Gradient Boosting":      "Gradient Boost.",
    "Quantum LSTM":           "Quantum LSTM",
    "VQC (Trained)":          "VQC (Trained)",
}
df["short"] = df["name"].map(SHORT)

# ── colour palette (light theme) ──
QUANTUM   = "#7c3aed"   # violet-600
QUANTUM_L = "#c4b5fd"   # violet-300  (light fill)
CLASSICAL = "#059669"   # emerald-600
CLASSIC_L = "#a7f3d0"   # emerald-200 (light fill)
OURS      = "#8b5cf6"   # violet-500  (highlight)
OURS_L    = "#ddd6fe"   # violet-200

RED       = "#ef4444"
BG_WHITE  = "#ffffff"
BG_LIGHT  = "#f9fafb"   # gray-50
GRID_CLR  = "#e5e7eb"   # gray-200
TEXT_DARK  = "#111827"   # gray-900
TEXT_MED   = "#374151"   # gray-700
TEXT_LIGHT = "#6b7280"   # gray-500
BORDER    = "#d1d5db"    # gray-300

def bar_fill(row):
    if row["is_ours"]: return OURS_L
    return QUANTUM_L if row["is_quantum"] else CLASSIC_L

def bar_edge(row):
    if row["is_ours"]: return OURS
    return QUANTUM if row["is_quantum"] else CLASSICAL

df["fill"]  = df.apply(bar_fill, axis=1)
df["edge"]  = df.apply(bar_edge, axis=1)
df["hatch"] = df["is_quantum"].map({True: "///", False: ""})

# ── global style (light theme) ──
plt.rcParams.update({
    "figure.facecolor":   BG_WHITE,
    "axes.facecolor":     BG_WHITE,
    "axes.edgecolor":     BORDER,
    "axes.labelcolor":    TEXT_DARK,
    "axes.grid":          True,
    "axes.linewidth":     0.8,
    "grid.color":         GRID_CLR,
    "grid.linewidth":     0.5,
    "grid.linestyle":     "--",
    "xtick.color":        TEXT_MED,
    "ytick.color":        TEXT_MED,
    "text.color":         TEXT_DARK,
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":          11,
    "legend.facecolor":   BG_WHITE,
    "legend.edgecolor":   BORDER,
    "legend.framealpha":  0.95,
    "savefig.facecolor":  BG_WHITE,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.3,
})

LEGEND_HANDLES = [
    Patch(facecolor=QUANTUM_L, edgecolor=QUANTUM, linewidth=1.2,
          label="Quantum", hatch="///"),
    Patch(facecolor=CLASSIC_L, edgecolor=CLASSICAL, linewidth=1.2,
          label="Classical"),
    Patch(facecolor=OURS_L, edgecolor=OURS, linewidth=1.5,
          label="★ QORC+Ridge (ours)", hatch="///"),
]


def add_legend(ax, loc="upper right", **kw):
    ax.legend(handles=LEGEND_HANDLES, loc=loc, fontsize=9, borderpad=0.8, **kw)


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════
#  FIG 1 — Horizontal bar chart: Latent MSE  (sorted best→worst)
# ═══════════════════════════════════════════════════════════════════
def fig_latent_mse():
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(df))[::-1]
    bars = ax.barh(
        y, df["latent_MSE"], height=0.65,
        color=df["fill"], edgecolor=df["edge"], linewidth=1.3,
        hatch=df["hatch"].tolist(), zorder=3,
    )
    for bar, val, is_ours in zip(bars, df["latent_MSE"], df["is_ours"]):
        weight = "bold" if is_ours else "medium"
        col = OURS if is_ours else TEXT_MED
        ax.text(
            bar.get_width() + 0.0008, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9, color=col, fontweight=weight,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(df["short"], fontsize=10)
    ax.set_xlabel("Latent MSE  (lower is better)", fontsize=11, fontweight="medium")
    ax.set_title("Benchmark: Latent-Space MSE", fontsize=16, fontweight="bold",
                 pad=16, color=TEXT_DARK)
    ax.set_xlim(0, df["latent_MSE"].max() * 1.20)
    ax.invert_yaxis()
    add_legend(ax, loc="lower right")
    save(fig, "01_latent_mse.png")


# ═══════════════════════════════════════════════════════════════════
#  FIG 2 — R² bar chart
# ═══════════════════════════════════════════════════════════════════
def fig_r2():
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(df))[::-1]
    bars = ax.barh(
        y, df["R2"], height=0.65,
        color=df["fill"], edgecolor=df["edge"], linewidth=1.3,
        hatch=df["hatch"].tolist(), zorder=3,
    )
    for bar, val, is_ours in zip(bars, df["R2"], df["is_ours"]):
        xpos = max(val, 0) + 0.01
        weight = "bold" if is_ours else "medium"
        col = OURS if is_ours else TEXT_MED
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color=col, fontweight=weight)
    ax.axvline(0, color=RED, linewidth=0.8, linestyle="--", alpha=0.5, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(df["short"], fontsize=10)
    ax.set_xlabel("R² Score  (higher is better)", fontsize=11, fontweight="medium")
    ax.set_title("Benchmark: R² Score", fontsize=16, fontweight="bold",
                 pad=16, color=TEXT_DARK)
    ax.set_xlim(min(df["R2"].min() - 0.06, -0.3), 1.0)
    ax.invert_yaxis()
    add_legend(ax, loc="lower right")
    save(fig, "02_r2_score.png")


# ═══════════════════════════════════════════════════════════════════
#  FIG 3 — Surface RMSE
# ═══════════════════════════════════════════════════════════════════
def fig_surface_rmse():
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(df))[::-1]
    bars = ax.barh(
        y, df["surface_RMSE"], height=0.65,
        color=df["fill"], edgecolor=df["edge"], linewidth=1.3,
        hatch=df["hatch"].tolist(), zorder=3,
    )
    for bar, val, is_ours in zip(bars, df["surface_RMSE"], df["is_ours"]):
        weight = "bold" if is_ours else "medium"
        col = OURS if is_ours else TEXT_MED
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color=col, fontweight=weight)
    ax.set_yticks(y)
    ax.set_yticklabels(df["short"], fontsize=10)
    ax.set_xlabel("Surface RMSE  (lower is better)", fontsize=11, fontweight="medium")
    ax.set_title("Benchmark: Price-Space RMSE", fontsize=16, fontweight="bold",
                 pad=16, color=TEXT_DARK)
    ax.set_xlim(0, df["surface_RMSE"].max() * 1.18)
    ax.invert_yaxis()
    add_legend(ax, loc="lower right")
    save(fig, "03_surface_rmse.png")


# ═══════════════════════════════════════════════════════════════════
#  FIG 4 — Accuracy vs Speed scatter (Pareto frontier)
# ═══════════════════════════════════════════════════════════════════
def fig_scatter():
    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in df.iterrows():
        is_ours = row["is_ours"]
        marker = "*" if is_ours else ("D" if row["is_quantum"] else "o")
        size = 240 if is_ours else 100
        zorder = 10 if is_ours else 5
        ec = row["edge"]
        fc = row["fill"] if not is_ours else OURS
        ax.scatter(
            row["inference_ms"], row["latent_MSE"],
            c=fc, edgecolors=ec, s=size, marker=marker,
            linewidths=1.5, zorder=zorder, alpha=0.9 if is_ours else 0.75,
        )
        name = row["short"]
        ha = "left"
        offset = (1.15, 0.0003)
        if row["inference_ms"] > 100:
            ha = "right"
            offset = (0.85, 0.0003)
        ax.annotate(
            name, xy=(row["inference_ms"], row["latent_MSE"]),
            xytext=(row["inference_ms"] * offset[0], row["latent_MSE"] + offset[1]),
            fontsize=8, color=TEXT_LIGHT, ha=ha, va="bottom",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference Latency (ms, log scale) -- slower", fontsize=11, fontweight="medium")
    ax.set_ylabel("better -- Latent MSE -- worse", fontsize=11, fontweight="medium")
    ax.set_title("Accuracy vs Inference Speed", fontsize=16, fontweight="bold",
                 pad=16, color=TEXT_DARK)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.2f}ms" if v < 1 else (f"{v:.0f}ms" if v < 1000 else f"{v/1000:.1f}s")
    ))

    # Pareto shaded zone (bottom-left is ideal)
    ax.fill_between([0.01, 2], [-0.01, -0.01], [0.016, 0.016],
                    color=CLASSIC_L, alpha=0.18, zorder=0)
    ax.text(0.08, 0.0153, "Ideal zone (fast + accurate)", fontsize=8,
            fontstyle="italic", color=CLASSICAL, alpha=0.8)

    add_legend(ax, loc="upper left")
    save(fig, "04_accuracy_vs_speed.png")


# ═══════════════════════════════════════════════════════════════════
#  FIG 5 — Radar chart: top models multi-metric
# ═══════════════════════════════════════════════════════════════════
def fig_radar():
    # Normalize metrics to [0, 1]
    def norm(arr, invert=False):
        mn, mx = arr.min(), arr.max()
        n = (arr - mn) / (mx - mn + 1e-12)
        return 1 - n if invert else n

    metrics = {
        "Latent Accuracy":   norm(df["latent_MSE"].values, invert=True),
        "R² Score":          norm(df["R2"].values, invert=False),
        "Surface Accuracy":  norm(df["surface_RMSE"].values, invert=True),
        "Fast Training":     norm(df["train_time"].values.clip(0.001), invert=True),
        "Fast Inference":    norm(df["inference_ms"].values, invert=True),
    }
    labels = list(metrics.keys())
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Models to show
    show = {
        "Simple PML + Ridge":     {"c": QUANTUM,   "ls": "-",  "lw": 2},
        "Ridge Regression":       {"c": CLASSICAL, "ls": "-",  "lw": 2},
        "★ QORC + Ridge (ours)":  {"c": OURS,      "ls": "-",  "lw": 2.5},
        "Quantum LSTM":           {"c": RED,       "ls": "--", "lw": 1.5},
    }

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor(BG_WHITE)
    fig.patch.set_facecolor(BG_WHITE)

    for name, style in show.items():
        idx = df.index[df["name"] == name][0]
        vals = [metrics[m][idx] for m in labels]
        vals += vals[:1]
        ax.plot(angles, vals, color=style["c"], linewidth=style["lw"],
                linestyle=style["ls"], label=SHORT[name])
        ax.fill(angles, vals, color=style["c"], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color=TEXT_MED)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], fontsize=0)
    ax.set_ylim(0, 1.05)
    ax.spines["polar"].set_color(GRID_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.5)

    ax.set_title("Multi-Metric Radar Comparison", fontsize=16, fontweight="bold",
                 pad=24, color=TEXT_DARK)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.12), fontsize=9,
              framealpha=0.95, edgecolor=BORDER)
    save(fig, "05_radar.png")


# ═══════════════════════════════════════════════════════════════════
#  FIG 6 — Quantum vs Classical grouped summary (3 metrics)
# ═══════════════════════════════════════════════════════════════════
def fig_quantum_vs_classical():
    # Top-3 quantum (exclude VQC & QLSTM — overfitting failures)
    q_good = df[df["is_quantum"] & ~df["name"].isin(["VQC (Trained)", "Quantum LSTM"])]
    c_all  = df[~df["is_quantum"]]

    panels = [
        ("Latent MSE (lower)",   "latent_MSE",   False),
        ("Surface RMSE (lower)", "surface_RMSE", False),
        ("R2 Score (higher)",     "R2",           True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Quantum (Top 3) vs Classical (All 6) — Average Performance",
                 fontsize=15, fontweight="bold", y=1.03, color=TEXT_DARK)

    for ax, (label, col, higher_better) in zip(axes, panels):
        qv = q_good[col].mean()
        cv = c_all[col].mean()

        bars = ax.bar(
            [0, 1], [qv, cv], width=0.5,
            color=[QUANTUM_L, CLASSIC_L],
            edgecolor=[QUANTUM, CLASSICAL],
            linewidth=1.5, hatch=["///", ""], zorder=3,
        )
        for bar, val in zip(bars, [qv, cv]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(bar.get_height()) * 0.025 + 0.001,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=12, color=TEXT_DARK, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Quantum\n(top 3)", "Classical\n(all 6)"],
                           fontsize=10, fontweight="medium")
        ax.set_title(label, fontsize=13, fontweight="bold", pad=10, color=TEXT_DARK)

        # Winner badge
        ymin = min(qv, cv); ymax = max(qv, cv)
        margin = (ymax - ymin) * 0.45 + 0.002
        ax.set_ylim(max(0, ymin - margin), ymax + margin)

        if (higher_better and qv > cv) or (not higher_better and qv < cv):
            winner, w_col = "Quantum wins", QUANTUM
        else:
            winner, w_col = "Classical wins", CLASSICAL
        ax.text(0.5, 0.02, winner, ha="center", va="bottom",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                color=w_col, alpha=0.85)

    fig.tight_layout()
    save(fig, "06_quantum_vs_classical.png")


# ═══════════════════════════════════════════════════════════════════
#  FIG 7 — Training efficiency: time vs MSE (bubble = params)
# ═══════════════════════════════════════════════════════════════════
def fig_training_efficiency():
    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in df.iterrows():
        t = max(row["train_time"], 0.005)
        is_ours = row["is_ours"]
        marker = "*" if is_ours else ("D" if row["is_quantum"] else "o")
        size = 220 if is_ours else 100
        fc = OURS if is_ours else (QUANTUM_L if row["is_quantum"] else CLASSIC_L)
        ec = row["edge"]
        ax.scatter(t, row["latent_MSE"], c=fc, edgecolors=ec,
                   s=size, marker=marker, linewidths=1.5,
                   zorder=10 if is_ours else 5, alpha=0.85)
        name = row["short"]
        ax.annotate(name, xy=(t, row["latent_MSE"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=7.5, color=TEXT_LIGHT)

    ax.set_xscale("log")
    ax.set_xlabel("Training Time (seconds, log scale)", fontsize=11, fontweight="medium")
    ax.set_ylabel("better -- Latent MSE -- worse", fontsize=11, fontweight="medium")
    ax.set_title("Training Efficiency: Time vs Accuracy",
                 fontsize=16, fontweight="bold", pad=16, color=TEXT_DARK)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.2f}s" if v < 1 else f"{v:.0f}s"
    ))
    add_legend(ax, loc="upper left")
    save(fig, "07_training_efficiency.png")


# ═══════════════════════════════════════════════════════════════════
#  FIG 8 — Model leaderboard overview (visual table)
# ═══════════════════════════════════════════════════════════════════
def fig_leaderboard():
    """Compact visual leaderboard: rank + MSE + R² + speed in one picture."""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis("off")

    # Table data
    headers = ["Rank", "Model", "Type", "Latent MSE", "Surf. RMSE", "R²", "Inference"]
    col_widths = [0.06, 0.19, 0.1, 0.13, 0.13, 0.1, 0.12]
    x_starts = [0]
    for w in col_widths[:-1]:
        x_starts.append(x_starts[-1] + w)

    y_top = 0.92
    row_h = 0.065

    # Header
    for j, (hdr, xs, w) in enumerate(zip(headers, x_starts, col_widths)):
        ax.text(xs + w / 2, y_top + 0.02, hdr, ha="center", va="center",
                fontsize=10, fontweight="bold", color=BG_WHITE,
                transform=ax.transAxes)
    # Header bg
    header_box = FancyBboxPatch(
        (0, y_top - 0.015), sum(col_widths), row_h * 0.85,
        boxstyle="round,pad=0.008", facecolor="#1e293b", edgecolor="none",
        transform=ax.transAxes, zorder=2,
    )
    ax.add_patch(header_box)

    # Rows
    for i, (_, row) in enumerate(df.iterrows()):
        y = y_top - (i + 1) * row_h - 0.015
        is_ours = row["is_ours"]
        is_q = row["is_quantum"]

        # Row background
        if is_ours:
            bg_color = (*matplotlib.colors.to_rgb(OURS_L), 0.35)
        elif i % 2 == 0:
            bg_color = (0.97, 0.97, 0.97, 1)
        else:
            bg_color = (1, 1, 1, 1)
        row_box = FancyBboxPatch(
            (0, y - row_h * 0.15), sum(col_widths), row_h * 0.85,
            boxstyle="round,pad=0.005", facecolor=bg_color, edgecolor=GRID_CLR,
            linewidth=0.5, transform=ax.transAxes, zorder=1,
        )
        ax.add_patch(row_box)

        # Cell values
        rank = str(i + 1)
        name = row["short"]
        typ = "Quantum" if is_q else "Classical"
        typ_color = QUANTUM if is_q else CLASSICAL
        mse = f"{row['latent_MSE']:.4f}"
        rmse = f"{row['surface_RMSE']:.4f}"
        r2 = f"{row['R2']:.3f}"
        inf_ms = row["inference_ms"]
        inf = f"{inf_ms:.2f}ms" if inf_ms < 1 else (f"{inf_ms:.1f}ms" if inf_ms < 100 else f"{inf_ms:.0f}ms")

        vals = [rank, name, typ, mse, rmse, r2, inf]
        colors = [TEXT_MED, OURS if is_ours else TEXT_DARK, typ_color,
                  TEXT_DARK, TEXT_DARK, TEXT_DARK, TEXT_DARK]
        weights = ["medium", "bold" if is_ours else "medium", "bold",
                   "medium", "medium", "medium", "medium"]

        for j, (v, xs, w, c, fw) in enumerate(zip(vals, x_starts, col_widths, colors, weights)):
            ax.text(xs + w / 2, y + row_h * 0.25, v,
                    ha="center", va="center", fontsize=9,
                    color=c, fontweight=fw, transform=ax.transAxes)

    ax.set_title("Full Benchmark Leaderboard",
                 fontsize=16, fontweight="bold", pad=20, color=TEXT_DARK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "08_leaderboard.png")


# ═══════════════════════════════════════════════════════════════════
#  FIG 9 — Swaption Surface Heatmaps (3-panel: first, mid, last day)
# ═══════════════════════════════════════════════════════════════════
def fig_swaption_surfaces():
    import sys
    sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
    from src.preprocessing import load_train_data, get_unique_tenors_maturities

    dates, columns, prices = load_train_data(
        os.path.join(SCRIPT_DIR, "..", "DATASETS", "train.xlsx")
    )
    tenors, maturities = get_unique_tenors_maturities(columns)

    # Pick 3 days: first, middle, last
    indices = [0, len(prices) // 2, len(prices) - 1]
    day_labels = ["First Day", "Mid-Point", "Last Day"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Swaption Surface Evolution Over Time",
                 fontsize=16, fontweight="bold", y=1.02, color=TEXT_DARK)

    vmin = prices[indices].min()
    vmax = prices[indices].max()

    for ax, idx, label in zip(axes, indices, day_labels):
        surface = prices[idx].reshape(len(tenors), len(maturities))
        im = ax.imshow(surface, aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax, origin="lower")
        ax.set_xticks(np.arange(0, len(maturities), 3))
        ax.set_xticklabels([f"{maturities[i]:.1f}" for i in range(0, len(maturities), 3)],
                           fontsize=8)
        ax.set_yticks(np.arange(0, len(tenors), 3))
        ax.set_yticklabels([f"{tenors[i]:.0f}" for i in range(0, len(tenors), 3)],
                           fontsize=8)
        ax.set_xlabel("Maturity (years)", fontsize=9)
        ax.set_ylabel("Tenor (years)", fontsize=9)
        date_str = str(dates[idx])[:10] if dates[idx] else f"Day {idx}"
        ax.set_title(f"{label}\n{date_str}", fontsize=11, fontweight="bold", pad=8)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Price", fontsize=10)
    fig.tight_layout()
    save(fig, "09_swaption_surfaces.png")


# ═══════════════════════════════════════════════════════════════════
#  FIG 10 — AE Reconstruction Quality
# ═══════════════════════════════════════════════════════════════════
def fig_ae_reconstruction():
    import sys, torch
    sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
    from src.preprocessing import load_train_data, SwaptionPreprocessor, get_unique_tenors_maturities
    from src.autoencoder import load_autoencoder

    ROOT = os.path.join(SCRIPT_DIR, "..")
    dates, columns, prices = load_train_data(os.path.join(ROOT, "DATASETS", "train.xlsx"))
    tenors, maturities = get_unique_tenors_maturities(columns)

    # Rebuild preprocessor from saved params
    prep = np.load(os.path.join(ROOT, "outputs", "preprocessor.npz"), allow_pickle=True)
    preprocessor = SwaptionPreprocessor()
    preprocessor.median_ = prep["median"]
    preprocessor.iqr_ = prep["iqr"]
    preprocessor.min_ = prep["min"]
    preprocessor.range_ = prep["range"]
    preprocessor.clip_lower_ = prep["clip_lower"]
    preprocessor.clip_upper_ = prep["clip_upper"]
    preprocessor.is_fitted = True

    prices_norm = preprocessor.transform(prices)

    ae = load_autoencoder(os.path.join(ROOT, "outputs", "ae_weights.pt"), device="cpu")
    with torch.no_grad():
        x = torch.tensor(prices_norm, dtype=torch.float32)
        x_hat, _ = ae(x, mask_ratio=0.0)
        recon = x_hat.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Autoencoder Reconstruction Quality",
                 fontsize=16, fontweight="bold", y=1.02, color=TEXT_DARK)

    # Panel 1: Original vs Reconstructed scatter
    ax = axes[0]
    # Subsample for clarity
    n_pts = min(5000, prices_norm.size)
    rng = np.random.default_rng(42)
    flat_orig = prices_norm.ravel()
    flat_recon = recon.ravel()
    idx = rng.choice(len(flat_orig), n_pts, replace=False)
    ax.scatter(flat_orig[idx], flat_recon[idx], s=4, alpha=0.3, c=QUANTUM, edgecolors="none")
    ax.plot([0, 1], [0, 1], "--", color=RED, linewidth=1.2, alpha=0.7, label="Perfect")
    rmse = np.sqrt(np.mean((prices_norm - recon) ** 2))
    ax.set_xlabel("Original (normalized)", fontsize=10)
    ax.set_ylabel("Reconstructed", fontsize=10)
    ax.set_title(f"Original vs Reconstructed (RMSE={rmse:.4f})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Panel 2: Reconstruction error heatmap (tenor × maturity, averaged over time)
    ax = axes[1]
    errors = np.mean((prices_norm - recon) ** 2, axis=0)  # (224,)
    error_grid = errors.reshape(len(tenors), len(maturities))
    im = ax.imshow(error_grid, aspect="auto", cmap="Reds", origin="lower")
    ax.set_xticks(np.arange(0, len(maturities), 3))
    ax.set_xticklabels([f"{maturities[i]:.1f}" for i in range(0, len(maturities), 3)], fontsize=8)
    ax.set_yticks(np.arange(0, len(tenors), 3))
    ax.set_yticklabels([f"{tenors[i]:.0f}" for i in range(0, len(tenors), 3)], fontsize=8)
    ax.set_xlabel("Maturity (years)", fontsize=10)
    ax.set_ylabel("Tenor (years)", fontsize=10)
    ax.set_title("Reconstruction Error Heatmap (MSE)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.85)

    fig.tight_layout()
    save(fig, "10_ae_reconstruction.png")


# ═══════════════════════════════════════════════════════════════════
#  FIG 11 — Latent Code Trajectories
# ═══════════════════════════════════════════════════════════════════
def fig_latent_trajectories():
    ROOT = os.path.join(SCRIPT_DIR, "..")
    latent = np.load(os.path.join(ROOT, "outputs", "latent_codes.npy"))  # (494, 20)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#7c3aed", "#059669", "#ef4444", "#f59e0b", "#3b82f6"]
    for i in range(5):
        ax.plot(latent[:, i], color=colors[i], alpha=0.75, linewidth=1.2,
                label=f"Dim {i+1}")
    ax.set_xlabel("Timestep", fontsize=11, fontweight="medium")
    ax.set_ylabel("Latent Value", fontsize=11, fontweight="medium")
    ax.set_title("Latent Code Trajectories (First 5 Dimensions)",
                 fontsize=16, fontweight="bold", pad=14, color=TEXT_DARK)
    ax.legend(fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "11_latent_trajectories.png")


# ═══════════════════════════════════════════════════════════════════
#  FIG 12 — Quantum Feature Distribution (3-panel, one per reservoir)
# ═══════════════════════════════════════════════════════════════════
def fig_quantum_feature_dist():
    ROOT = os.path.join(SCRIPT_DIR, "..")
    qf = np.load(os.path.join(ROOT, "outputs", "quantum_features.npy"))  # (489, 1215)

    # Reservoir splits: R1=12m/3ph→364, R2=10m/4ph→715, R3=16m/2ph→136
    splits = [
        ("R1: 12 modes, 3 photons\n(364 features)", 0, 364, "#7c3aed"),
        ("R2: 10 modes, 4 photons\n(715 features)", 364, 1079, "#059669"),
        ("R3: 16 modes, 2 photons\n(136 features)", 1079, 1215, "#3b82f6"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Quantum Feature Distributions by Reservoir",
                 fontsize=16, fontweight="bold", y=1.03, color=TEXT_DARK)

    for ax, (label, s, e, color) in zip(axes, splits):
        data = qf[:, s:e].ravel()
        ax.hist(data, bins=80, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.set_title(label, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Fock Probability", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.text(0.95, 0.92, f"μ={data.mean():.4f}\nσ={data.std():.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=BORDER, alpha=0.9))

    fig.tight_layout()
    save(fig, "12_quantum_feature_dist.png")


# ═══════════════════════════════════════════════════════════════════
#  FIG 13 — Prediction vs Ground Truth (Validation)
# ═══════════════════════════════════════════════════════════════════
def fig_prediction_vs_truth():
    import joblib
    ROOT = os.path.join(SCRIPT_DIR, "..")
    latent = np.load(os.path.join(ROOT, "outputs", "latent_codes.npy"))  # (494, 20)
    qf = np.load(os.path.join(ROOT, "outputs", "quantum_features.npy"))  # (489, 1215)
    ridge = joblib.load(os.path.join(ROOT, "outputs", "ridge_model.joblib"))

    # Reconstruct the feature matrix used for training (window=5)
    window = 5
    n_samples = len(qf)  # 489

    # Build classical context: window of latent codes + delta
    X_classical = []
    for t in range(window, window + n_samples):
        window_flat = latent[t - window:t].ravel()  # 5*20 = 100
        delta = latent[t - 1] - latent[t - 2]  # 20
        X_classical.append(np.concatenate([window_flat, delta]))
    X_classical = np.array(X_classical)  # (489, 120)

    X_full = np.concatenate([qf, X_classical], axis=1)  # (489, 1335)
    y_true = latent[window:]  # (489, 20)

    y_pred = ridge.predict(X_full)

    # Use last 50 as validation
    val_size = 50
    y_true_val = y_true[-val_size:]
    y_pred_val = y_pred[-val_size:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Prediction vs Ground Truth (Validation Set)",
                 fontsize=16, fontweight="bold", y=1.02, color=TEXT_DARK)

    # Panel 1: 3 sample latent dimensions over validation timesteps
    ax = axes[0]
    dims = [0, 5, 15]
    colors_p = ["#7c3aed", "#059669", "#ef4444"]
    t_axis = np.arange(val_size)
    for dim, c in zip(dims, colors_p):
        ax.plot(t_axis, y_true_val[:, dim], color=c, linewidth=1.5, label=f"Actual (dim {dim})")
        ax.plot(t_axis, y_pred_val[:, dim], color=c, linewidth=1.5, linestyle="--",
                alpha=0.7, label=f"Predicted (dim {dim})")
    ax.set_xlabel("Validation Timestep", fontsize=10)
    ax.set_ylabel("Latent Value", fontsize=10)
    ax.set_title("Predicted vs Actual (3 Latent Dims)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right")

    # Panel 2: Prediction error distribution
    ax = axes[1]
    errors = (y_pred_val - y_true_val).ravel()
    ax.hist(errors, bins=60, color=QUANTUM_L, edgecolor=QUANTUM, linewidth=0.5, alpha=0.8)
    ax.axvline(0, color=RED, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xlabel("Prediction Error", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Error Distribution (μ={errors.mean():.4f}, σ={errors.std():.4f})",
                 fontsize=12, fontweight="bold")

    fig.tight_layout()
    save(fig, "13_prediction_vs_truth.png")


# ═══════════════════════════════════════════════════════════════════
#  FIG 14 — Quantum Feature Importance (Ridge Coefficients)
# ═══════════════════════════════════════════════════════════════════
def fig_feature_importance():
    import joblib
    ROOT = os.path.join(SCRIPT_DIR, "..")
    ridge = joblib.load(os.path.join(ROOT, "outputs", "ridge_model.joblib"))
    coef = np.abs(ridge.coef_)  # (20, 1335)

    # Average absolute coefficient across all 20 output dims
    avg_coef = coef.mean(axis=0)  # (1335,)

    # Quantum features: 0:1215, Classical: 1215:1335
    q_importance = avg_coef[:1215].mean()
    c_importance = avg_coef[1215:].mean()

    # Per-reservoir breakdown
    r1 = avg_coef[:364].mean()
    r2 = avg_coef[364:1079].mean()
    r3 = avg_coef[1079:1215].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Feature Importance Analysis (Ridge Coefficients)",
                 fontsize=16, fontweight="bold", y=1.02, color=TEXT_DARK)

    # Panel 1: Quantum vs Classical
    ax = axes[0]
    bars = ax.bar(
        ["Quantum\nFeatures\n(1,215 dims)", "Classical\nContext\n(120 dims)"],
        [q_importance, c_importance],
        color=[QUANTUM_L, CLASSIC_L], edgecolor=[QUANTUM, CLASSICAL],
        linewidth=1.5, width=0.5, zorder=3
    )
    for bar, val in zip(bars, [q_importance, c_importance]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00002,
                f"{val:.5f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Quantum vs Classical Feature Weight", fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg |coefficient|", fontsize=10)

    # Panel 2: Per-reservoir breakdown
    ax = axes[1]
    labels = ["R1\n12m/3ph\n(364)", "R2\n10m/4ph\n(715)", "R3\n16m/2ph\n(136)", "Classical\n(120)"]
    vals = [r1, r2, r3, c_importance]
    colors_b = ["#7c3aed", "#059669", "#3b82f6", CLASSIC_L]
    edges_b = ["#5b21b6", "#047857", "#1d4ed8", CLASSICAL]
    bars = ax.bar(labels, vals, color=colors_b, edgecolor=edges_b, linewidth=1.5, width=0.55, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00002,
                f"{val:.5f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Per-Reservoir Feature Weight", fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg |coefficient|", fontsize=10)

    fig.tight_layout()
    save(fig, "14_feature_importance.png")


# ═══════════════════════════════════════════════════════════════════
#  JSON Export for Plotly.js (website interactive charts)
# ═══════════════════════════════════════════════════════════════════
def export_json():
    import json, joblib, sys
    sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
    from src.preprocessing import load_train_data, get_unique_tenors_maturities

    ROOT = os.path.join(SCRIPT_DIR, "..")
    ASSETS = os.path.join(ROOT, "website", "assets")
    os.makedirs(ASSETS, exist_ok=True)

    # ── surface_data.json ──
    dates, columns, prices = load_train_data(
        os.path.join(ROOT, "DATASETS", "train.xlsx")
    )
    tenors, maturities = get_unique_tenors_maturities(columns)
    indices = [0, len(prices) // 2, len(prices) - 1]
    surfaces = []
    for idx in indices:
        surface = prices[idx].reshape(len(tenors), len(maturities))
        surfaces.append({
            "date": str(dates[idx])[:10],
            "values": surface.tolist(),
        })
    surface_json = {
        "tenors": tenors,
        "maturities": maturities,
        "surfaces": surfaces,
    }
    with open(os.path.join(ASSETS, "surface_data.json"), "w") as f:
        json.dump(surface_json, f)
    print(f"  ✓ {ASSETS}/surface_data.json")

    # ── latent_trajectories.json ──
    latent = np.load(os.path.join(ROOT, "outputs", "latent_codes.npy"))
    latent_json = {
        "timesteps": list(range(len(latent))),
        "dimensions": {}
    }
    for i in range(5):
        latent_json["dimensions"][f"dim_{i+1}"] = latent[:, i].tolist()
    with open(os.path.join(ASSETS, "latent_trajectories.json"), "w") as f:
        json.dump(latent_json, f)
    print(f"  ✓ {ASSETS}/latent_trajectories.json")

    # ── prediction_comparison.json ──
    qf = np.load(os.path.join(ROOT, "outputs", "quantum_features.npy"))
    ridge = joblib.load(os.path.join(ROOT, "outputs", "ridge_model.joblib"))
    window = 5
    n_samples = len(qf)
    X_classical = []
    for t in range(window, window + n_samples):
        window_flat = latent[t - window:t].ravel()
        delta = latent[t - 1] - latent[t - 2]
        X_classical.append(np.concatenate([window_flat, delta]))
    X_classical = np.array(X_classical)
    X_full = np.concatenate([qf, X_classical], axis=1)
    y_true = latent[window:]
    y_pred = ridge.predict(X_full)
    val_size = 50

    pred_json = {
        "timesteps": list(range(val_size)),
        "columns": {}
    }
    for dim in [0, 5, 15]:
        pred_json["columns"][f"dim_{dim}"] = {
            "actual": y_true[-val_size:, dim].tolist(),
            "predicted": y_pred[-val_size:, dim].tolist(),
        }
    with open(os.path.join(ASSETS, "prediction_comparison.json"), "w") as f:
        json.dump(pred_json, f)
    print(f"  ✓ {ASSETS}/prediction_comparison.json")

    # ── benchmark_results.json ──
    results = pd.read_csv(CSV_PATH)
    bench_json = {
        "columns": results.columns.tolist(),
        "data": results.values.tolist(),
    }
    with open(os.path.join(ASSETS, "benchmark_results.json"), "w") as f:
        json.dump(bench_json, f)
    print(f"  ✓ {ASSETS}/benchmark_results.json")


# ═══════════════════════════════════════════════════════════════════
#  Run all
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating benchmark figures (light theme)...")
    print(f"  Source: {CSV_PATH}")
    print(f"  Output: {FIG_DIR}/\n")

    # Original 8 figures
    fig_latent_mse()
    fig_r2()
    fig_surface_rmse()
    fig_scatter()
    fig_radar()
    fig_quantum_vs_classical()
    fig_training_efficiency()
    fig_leaderboard()

    # New figures (9-14)
    print("\nGenerating new insight figures...")
    fig_swaption_surfaces()
    fig_ae_reconstruction()
    fig_latent_trajectories()
    fig_quantum_feature_dist()
    fig_prediction_vs_truth()
    fig_feature_importance()

    # JSON export for website
    print("\nExporting JSON data for Plotly.js...")
    export_json()

    print(f"\nDone — 14 figures saved to {FIG_DIR}/")
    print(f"JSON assets exported to website/assets/")
