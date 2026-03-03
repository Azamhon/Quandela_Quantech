#!/usr/bin/env python
"""
Generate all figures from quantech.ipynb as static PNGs for the website.
Saves to website/figures/
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_train_data, SwaptionPreprocessor
from src.autoencoder import load_autoencoder
from src.hybrid_model import make_windows
from src.quantum_reservoir import (
    EnsembleQORC, extract_quantum_features, QuantumFeatureNormalizer,
)
from src.utils import load_config, set_seed
from sklearn.linear_model import Ridge
from math import comb

OUT = os.path.join(PROJECT_ROOT, "website", "figures")
os.makedirs(OUT, exist_ok=True)

cfg = load_config(os.path.join(PROJECT_ROOT, "configs/config.yaml"))
set_seed(cfg["seed"])

# ── Load data ────────────────────────────────────────────────
print("Loading data...")
dates, price_columns, prices_raw = load_train_data(
    os.path.join(PROJECT_ROOT, cfg["data"]["train_path"])
)
preprocessor = SwaptionPreprocessor(
    winsorize_limits=tuple(cfg["preprocessing"]["winsorize_limits"])
)
prices_norm = preprocessor.fit_transform(prices_raw)

# ── Load AE ──────────────────────────────────────────────────
ae_cfg = cfg["autoencoder"]
ae_model = load_autoencoder(
    os.path.join(PROJECT_ROOT, "outputs/ae_weights.pt"),
    input_dim=ae_cfg["input_dim"],
    hidden_dims=tuple(ae_cfg["hidden_dims"]),
    latent_dim=ae_cfg["latent_dim"],
    device="cpu",
)
ae_model.eval()
for p in ae_model.parameters():
    p.requires_grad_(False)

with torch.no_grad():
    x_tensor = torch.tensor(prices_norm, dtype=torch.float32)
    x_hat, z = ae_model(x_tensor)
    latent_codes = z.numpy()
    reconstructed = x_hat.numpy()

# ── Splits ───────────────────────────────────────────────────
window_size = cfg["hybrid_model"]["window_size"]
val_split = cfg["autoencoder"]["val_split"]
X_all, y_all, indices = make_windows(latent_codes, window_size=window_size)
n_total = len(X_all)
n_train = n_total - val_split
X_train, X_val = X_all[:n_train], X_all[n_train:]
y_train, y_val = y_all[:n_train], y_all[n_train:]

Q_all = np.load(os.path.join(PROJECT_ROOT, "outputs/quantum_features.npy"))
Q_train, Q_val = Q_all[:n_train], Q_all[n_train:]

# ── Train Ridge ──────────────────────────────────────────────
X_tr_full = np.hstack([Q_train, X_train])
X_vl_full = np.hstack([Q_val, X_val])
ridge = Ridge(alpha=100.0)
ridge.fit(X_tr_full, y_train)
val_preds = ridge.predict(X_vl_full)

# ── Test data ────────────────────────────────────────────────
_, _, test_prices_raw = load_train_data(
    os.path.join(PROJECT_ROOT, "DATASETS/test.xlsx")
)
test_prices_norm = preprocessor.transform(test_prices_raw)
with torch.no_grad():
    test_latent = ae_model.encode(
        torch.tensor(test_prices_norm, dtype=torch.float32)
    ).numpy()

all_latent = np.concatenate([latent_codes, test_latent], axis=0)
X_all_ext, y_all_ext, _ = make_windows(all_latent, window_size=window_size)
n_test = test_prices_raw.shape[0]
X_test = X_all_ext[-n_test:]
y_test = y_all_ext[-n_test:]

# Quantum features for test
set_seed(cfg["seed"])
ensemble_configs = cfg.get("quantum_reservoir", {}).get("ensemble", [
    {"n_modes": 12, "n_photons": 3, "seed": 42},
    {"n_modes": 10, "n_photons": 4, "seed": 43},
    {"n_modes": 16, "n_photons": 2, "seed": 44},
])
ensemble = EnsembleQORC(
    input_dim=X_test.shape[1], configs=ensemble_configs,
    use_fock=cfg.get("quantum_reservoir", {}).get("use_fock", True), device="cpu",
)
ensemble.eval()
Q_test_raw = extract_quantum_features(
    ensemble, torch.tensor(X_test, dtype=torch.float32), batch_size=64
)
Q_train_raw = extract_quantum_features(
    ensemble, torch.tensor(np.concatenate([X_train, X_val], axis=0), dtype=torch.float32),
    batch_size=64,
)
normalizer = QuantumFeatureNormalizer()
_ = normalizer.fit_transform(Q_train_raw)
Q_test = normalizer.transform(Q_test_raw)

X_test_full = np.hstack([Q_test, X_test])
test_preds = ridge.predict(X_test_full)
with torch.no_grad():
    test_surf_pred = ae_model.decode(torch.tensor(test_preds, dtype=torch.float32)).numpy()
    test_surf_true = ae_model.decode(torch.tensor(y_test, dtype=torch.float32)).numpy()
    surf_pred_val = ae_model.decode(torch.tensor(val_preds, dtype=torch.float32)).numpy()
    surf_true_val = ae_model.decode(torch.tensor(y_val, dtype=torch.float32)).numpy()

# ── Load benchmark results ───────────────────────────────────
results = pd.read_csv(os.path.join(PROJECT_ROOT, "benchmarks/results.csv"))
results_sorted = results.sort_values("test_latent_MSE").reset_index(drop=True)

QUANTUM_NAMES = {
    "Simple PML + Ridge", "★ QORC + Ridge (ours)",
    "QUANTECH MLP (ours)", "Quantum LSTM", "VQC (Trained)",
}
VIOLET = "#7c3aed"; VIOLET_L = "#c4b5fd"
EMERALD = "#059669"; EMERALD_L = "#a7f3d0"
OURS_C = "#8b5cf6"; OURS_L = "#ddd6fe"

SHORT = {
    "Simple PML + Ridge": "Simple PML+Ridge",
    "Ridge Regression": "Ridge",
    "sklearn MLP": "sklearn MLP",
    "★ QORC + Ridge (ours)": "QORC+Ridge (ours)",
    "Classical LSTM": "Classical LSTM",
    "Random Forest": "Random Forest",
    "QUANTECH MLP (ours)": "QUANTECH MLP",
    "SVR (RBF)": "SVR (RBF)",
    "Gradient Boosting": "Gradient Boost.",
    "Quantum LSTM": "Quantum LSTM",
    "VQC (Trained)": "VQC (Trained)",
}
df = results_sorted.copy()
df["short"] = df["name"].map(SHORT)
df["is_quantum"] = df["name"].isin(QUANTUM_NAMES)
df["is_ours"] = df["name"] == "★ QORC + Ridge (ours)"

def get_colors(row):
    if row["is_ours"]:
        return OURS_L, OURS_C
    elif row["is_quantum"]:
        return VIOLET_L, VIOLET
    else:
        return EMERALD_L, EMERALD

df["fill"] = df.apply(lambda r: get_colors(r)[0], axis=1)
df["edge"] = df.apply(lambda r: get_colors(r)[1], axis=1)

legend_patches = [
    Patch(facecolor=OURS_L, edgecolor=OURS_C, label="Ours (Quantum)"),
    Patch(facecolor=VIOLET_L, edgecolor=VIOLET, label="Other Quantum"),
    Patch(facecolor=EMERALD_L, edgecolor=EMERALD, label="Classical"),
]


def save(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {name}")


# ═══════════════════════════════════════════════════════════
# FIGURE 1: Swaption Surface Heatmap + Time Series
# ═══════════════════════════════════════════════════════════
print("Fig 1: Swaption surfaces...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im = axes[0].imshow(prices_raw, aspect="auto", cmap="viridis", interpolation="nearest")
axes[0].set_xlabel("Price dimension (224)")
axes[0].set_ylabel("Day index")
axes[0].set_title("Raw Swaption Surfaces (494 days)")
plt.colorbar(im, ax=axes[0], shrink=0.8)

dims = [0, 55, 111, 167, 223]
for d in dims:
    axes[1].plot(prices_raw[:, d], alpha=0.7, linewidth=0.8, label=f"dim {d}")
axes[1].set_xlabel("Day")
axes[1].set_ylabel("Price")
axes[1].set_title("Selected Price Dimensions Over Time")
axes[1].legend(fontsize=8, ncol=2)
plt.tight_layout()
save("01_swaption_surfaces.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 2: Before/After Preprocessing
# ═══════════════════════════════════════════════════════════
print("Fig 2: Preprocessing...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(prices_raw.ravel(), bins=100, alpha=0.7, color="#6366f1", edgecolor="white")
axes[0].set_title("Before Preprocessing")
axes[0].set_xlabel("Price")
axes[0].set_ylabel("Frequency")
axes[1].hist(prices_norm.ravel(), bins=100, alpha=0.7, color="#8b5cf6", edgecolor="white")
axes[1].set_title("After Preprocessing (normalised to [0,1])")
axes[1].set_xlabel("Normalised price")
plt.tight_layout()
save("02_preprocessing.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 3: AE Reconstruction
# ═══════════════════════════════════════════════════════════
print("Fig 3: AE reconstruction...")
np.random.seed(42)
sample_days = sorted(np.random.choice(prices_norm.shape[0], 4, replace=False))
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, day in zip(axes.ravel(), sample_days):
    ax.plot(prices_norm[day], linewidth=1, alpha=0.8, label="Original", color="#374151")
    ax.plot(reconstructed[day], linewidth=1, alpha=0.8, label="Reconstructed",
            color="#8b5cf6", linestyle="--")
    ax.set_title(f"Day {day}", fontsize=10)
    ax.set_xlabel("Price dimension")
    ax.legend(fontsize=8)
plt.suptitle("AE Reconstruction: Original vs Decoded", fontsize=12, fontweight="bold")
plt.tight_layout()
save("03_ae_reconstruction.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 4: Latent Space Analysis
# ═══════════════════════════════════════════════════════════
print("Fig 4: Latent space...")
fig, axes = plt.subplots(2, 1, figsize=(14, 7))
variances = latent_codes.var(axis=0)
top_dims = np.argsort(variances)[::-1][:6]
for d in top_dims:
    axes[0].plot(latent_codes[:, d], linewidth=0.8, alpha=0.8,
                 label=f"z[{d}] (var={variances[d]:.3f})")
axes[0].set_xlabel("Day")
axes[0].set_ylabel("Latent value")
axes[0].set_title("Temporal Dynamics of Top-Variance Latent Dimensions")
axes[0].legend(fontsize=7, ncol=3, loc="upper right")

corr = np.corrcoef(latent_codes.T)
im = axes[1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
axes[1].set_xlabel("Latent dim")
axes[1].set_ylabel("Latent dim")
axes[1].set_title("Latent Dimension Correlation Matrix")
plt.colorbar(im, ax=axes[1], shrink=0.8)
plt.tight_layout()
save("04_latent_analysis.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 5: Quantum Feature Analysis
# ═══════════════════════════════════════════════════════════
print("Fig 5: Quantum features...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(Q_all.ravel(), bins=100, alpha=0.7, color="#7c3aed", edgecolor="white")
axes[0].set_title("Quantum Feature Distribution")
axes[0].set_xlabel("Feature value")
axes[0].set_ylabel("Frequency")
axes[0].set_yscale("log")

q_var = Q_all.var(axis=0)
axes[1].plot(sorted(q_var, reverse=True), color="#7c3aed", linewidth=0.8)
axes[1].set_title("Quantum Feature Variance (sorted)")
axes[1].set_xlabel("Feature rank")
axes[1].set_ylabel("Variance")

im = axes[2].imshow(Q_all[:, :50].T, aspect="auto", cmap="magma")
axes[2].set_xlabel("Window index")
axes[2].set_ylabel("Feature index")
axes[2].set_title("Quantum Features (first 50 dims)")
plt.colorbar(im, ax=axes[2], shrink=0.8)
plt.tight_layout()
save("05_quantum_features.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 6: Predicted vs True Latent Codes (Validation)
# ═══════════════════════════════════════════════════════════
print("Fig 6: Val predictions...")
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for i in range(4):
    ax = axes[i // 2, i % 2]
    ax.plot(y_val[:, i], label="True", color="#374151", linewidth=1)
    ax.plot(val_preds[:, i], label="Predicted", color="#8b5cf6", linewidth=1, linestyle="--")
    ax.set_title(f"Latent dim {i}", fontsize=10)
    ax.set_xlabel("Val sample")
    ax.legend(fontsize=8)
plt.suptitle("QORC + Ridge: Predicted vs True Latent Codes (Validation)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save("06_val_predictions.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 7: Test Predictions — Surfaces
# ═══════════════════════════════════════════════════════════
print("Fig 7: Test predictions...")
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for i, ax in enumerate(axes.ravel()):
    if i >= n_test:
        ax.set_visible(False)
        continue
    ax.plot(test_surf_true[i], linewidth=1, alpha=0.8, label="Actual", color="#374151")
    ax.plot(test_surf_pred[i], linewidth=1, alpha=0.8, label="Predicted",
            color="#8b5cf6", linestyle="--")
    day_err = np.sqrt(np.mean((test_surf_true[i] - test_surf_pred[i])**2))
    ax.set_title(f"Test Day {i+1}  (RMSE={day_err:.4f})", fontsize=10)
    ax.set_xlabel("Price dim")
    ax.legend(fontsize=8)
plt.suptitle("QORC + Ridge: Predicted vs Actual Swaption Surfaces (6 Test Days)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save("07_test_predictions.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 8: Benchmark — Test Latent MSE
# ═══════════════════════════════════════════════════════════
print("Fig 8: Test Latent MSE...")
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(df)), df["test_latent_MSE"],
               color=df["fill"].values, edgecolor=df["edge"].values, linewidth=1.5)
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df["short"].values, fontsize=9)
ax.set_xlabel("Test Latent MSE (lower is better)")
ax.set_title("Benchmark: Test Latent MSE", fontsize=13, fontweight="bold")
ax.invert_yaxis()
for bar, val in zip(bars, df["test_latent_MSE"]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)
ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
plt.tight_layout()
save("08_test_latent_mse.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 9: Benchmark — Test Surface RMSE
# ═══════════════════════════════════════════════════════════
print("Fig 9: Test Surface RMSE...")
df_srmse = df.sort_values("test_surface_RMSE").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(df_srmse)), df_srmse["test_surface_RMSE"],
               color=df_srmse["fill"].values, edgecolor=df_srmse["edge"].values, linewidth=1.5)
ax.set_yticks(range(len(df_srmse)))
ax.set_yticklabels(df_srmse["short"].values, fontsize=9)
ax.set_xlabel("Test Surface RMSE (lower is better)")
ax.set_title("Benchmark: Test Surface RMSE (What Matters for Pricing)",
             fontsize=13, fontweight="bold")
ax.invert_yaxis()
for bar, val in zip(bars, df_srmse["test_surface_RMSE"]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)
ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
plt.tight_layout()
save("09_test_surface_rmse.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 10: Benchmark — Test R²
# ═══════════════════════════════════════════════════════════
print("Fig 10: Test R²...")
df_r2 = df.sort_values("test_R2", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(df_r2)), df_r2["test_R2"],
               color=df_r2["fill"].values, edgecolor=df_r2["edge"].values, linewidth=1.5)
ax.set_yticks(range(len(df_r2)))
ax.set_yticklabels(df_r2["short"].values, fontsize=9)
ax.set_xlabel("Test R² (higher is better)")
ax.set_title("Benchmark: Test R²", fontsize=13, fontweight="bold")
ax.invert_yaxis()
ax.axvline(0, color="#ef4444", linestyle="--", linewidth=0.8, alpha=0.5)
for bar, val in zip(bars, df_r2["test_R2"]):
    x_pos = max(bar.get_width(), 0) + 0.02
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8)
ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
plt.tight_layout()
save("10_test_r2.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 11: Accuracy vs Speed
# ═══════════════════════════════════════════════════════════
print("Fig 11: Accuracy vs speed...")
fig, ax = plt.subplots(figsize=(10, 7))
for _, row in df.iterrows():
    c = OURS_C if row["is_ours"] else (VIOLET if row["is_quantum"] else EMERALD)
    marker = "*" if row["is_ours"] else ("D" if row["is_quantum"] else "o")
    size = 200 if row["is_ours"] else 80
    ax.scatter(row["inference_ms"], row["test_latent_MSE"],
               color=c, s=size, marker=marker, edgecolors="white", linewidths=0.5, zorder=5)
    ax.annotate(row["short"], (row["inference_ms"], row["test_latent_MSE"]),
                fontsize=7, ha="left", va="bottom", xytext=(5, 3), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("Inference Latency (ms, log scale)")
ax.set_ylabel("Test Latent MSE (lower is better)")
ax.set_title("Accuracy vs Speed Trade-off", fontsize=13, fontweight="bold")
ax.legend(handles=legend_patches, fontsize=9)
plt.tight_layout()
save("11_accuracy_vs_speed.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 12: Val vs Test Generalisation
# ═══════════════════════════════════════════════════════════
print("Fig 12: Val vs Test...")
fig, ax = plt.subplots(figsize=(8, 8))
for _, row in df.iterrows():
    c = OURS_C if row["is_ours"] else (VIOLET if row["is_quantum"] else EMERALD)
    marker = "*" if row["is_ours"] else ("D" if row["is_quantum"] else "o")
    size = 200 if row["is_ours"] else 80
    ax.scatter(row["latent_MSE"], row["test_latent_MSE"],
               color=c, s=size, marker=marker, edgecolors="white", linewidths=0.5, zorder=5)
    ax.annotate(row["short"], (row["latent_MSE"], row["test_latent_MSE"]),
                fontsize=7, ha="left", va="bottom", xytext=(4, 3), textcoords="offset points")
lims = [0, max(df["latent_MSE"].max(), df["test_latent_MSE"].max()) * 1.1]
ax.plot(lims, lims, "--", color="#9ca3af", linewidth=1, alpha=0.7, label="Perfect generalisation")
ax.set_xlabel("Validation Latent MSE")
ax.set_ylabel("Test Latent MSE")
ax.set_title("Validation vs Test: Generalisation Check", fontsize=13, fontweight="bold")
ax.legend(handles=legend_patches + [plt.Line2D([0],[0], linestyle="--", color="#9ca3af",
          label="y=x")], fontsize=9)
ax.set_aspect("equal")
plt.tight_layout()
save("12_val_vs_test.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 13: Training Time
# ═══════════════════════════════════════════════════════════
print("Fig 13: Training time...")
df_time = df.sort_values("train_time", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(df_time)), df_time["train_time"],
               color=df_time["fill"].values, edgecolor=df_time["edge"].values, linewidth=1.5)
ax.set_yticks(range(len(df_time)))
ax.set_yticklabels(df_time["short"].values, fontsize=9)
ax.set_xlabel("Training Time (seconds)")
ax.set_title("Training Time Comparison", fontsize=13, fontweight="bold")
ax.invert_yaxis()
for bar, val in zip(bars, df_time["train_time"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}s", va="center", fontsize=8)
ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
plt.tight_layout()
save("13_training_time.png")


print(f"\nAll {len(os.listdir(OUT))} figures saved to {OUT}/")
