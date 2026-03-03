#!/usr/bin/env python3
"""Generate the quantech.ipynb notebook programmatically."""
import json, os, textwrap

def md(source):
    """Create a markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(True)}

def code(source):
    """Create a code cell."""
    return {"cell_type": "code", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(True), "outputs": [], "execution_count": None}

cells = []

# ═══════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════
cells.append(md("""
    # QUANTECH — Hybrid Photonic Quantum Reservoir Computing for Swaption Pricing

    > **Q-volution Hackathon 2026 · Quandela Challenge**
    >
    > *Team Quantech*

    ---

    This notebook provides a **comprehensive, end-to-end walkthrough** of our
    solution: a hybrid classical-quantum pipeline that uses **Perceval / MerLin
    photonic circuits** as a fixed quantum reservoir to predict swaption price
    surfaces.

    ### Pipeline Overview

    ```
    Raw Swaption Data (494×224)
      → Robust Preprocessing (winsorize → robust-scale → MinMax)
      → Sparse Denoising Autoencoder (224 → 20 latent dims)
      → Temporal Windowing (context window = 5 days)
      → Ensemble Photonic QRC (3 circuits → 1,215 Fock features)
      → Ridge Readout → Predicted latent code (20-dim)
      → AE Decoder → Predicted swaption surface (224 prices)
    ```

    ### Table of Contents

    | # | Section |
    |---|---------|
    | 1 | [Setup & Imports](#1-setup) |
    | 2 | [Data Loading & Exploration](#2-data) |
    | 3 | [Preprocessing Pipeline](#3-preprocessing) |
    | 4 | [Sparse Denoising Autoencoder](#4-autoencoder) |
    | 5 | [Latent Space Analysis](#5-latent) |
    | 6 | [Photonic Quantum Reservoir](#6-quantum) |
    | 7 | [Hybrid Model & Ridge Readout](#7-model) |
    | 8 | [Test Predictions (6 Future Days)](#8-test) |
    | 9 | [Benchmark Comparison (11 Models)](#9-benchmarks) |
    | 10 | [Conclusion & Quantum Advantage](#10-conclusion) |
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — SETUP
# ═══════════════════════════════════════════════════════════════
cells.append(md("""
    ---
    <a id="1-setup"></a>
    ## 1 · Setup & Imports
"""))

cells.append(code("""
    import os, sys, warnings
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch

    warnings.filterwarnings("ignore")
    %matplotlib inline
    plt.rcParams.update({
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })

    # Project paths
    PROJECT_ROOT = os.path.abspath(".")
    sys.path.insert(0, PROJECT_ROOT)

    from src.preprocessing import load_train_data, SwaptionPreprocessor
    from src.autoencoder import SparseDenosingAE, load_autoencoder
    from src.hybrid_model import make_windows, ClassicalHead
    from src.quantum_reservoir import (
        EnsembleQORC, extract_quantum_features,
        QuantumFeatureNormalizer, fock_output_size,
    )
    from src.utils import load_config, set_seed

    cfg = load_config("configs/config.yaml")
    set_seed(cfg["seed"])
    device = torch.device("cpu")
    print("Setup complete.")
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — DATA
# ═══════════════════════════════════════════════════════════════
cells.append(md("""
    ---
    <a id="2-data"></a>
    ## 2 · Data Loading & Exploration

    The dataset contains **494 daily swaption volatility surfaces**, each with
    **224 price points** across a tenor × maturity grid. These are European
    swaption prices observed over roughly two years.
"""))

cells.append(code("""
    # Load raw training data
    dates, price_columns, prices_raw = load_train_data(
        os.path.join(PROJECT_ROOT, cfg["data"]["train_path"])
    )

    print(f"Date range   : {dates[0]}  →  {dates[-1]}")
    print(f"Time steps   : {prices_raw.shape[0]}")
    print(f"Price points : {prices_raw.shape[1]}")
    print(f"Value range  : [{prices_raw.min():.4f}, {prices_raw.max():.4f}]")
    print(f"Has NaN      : {np.isnan(prices_raw).any()}")
"""))

cells.append(md("""
    ### 2.1 · Swaption Surface Heatmap

    Each row is a day; each column is a tenor × maturity combination.
    The heatmap reveals the smooth but evolving term structure.
"""))

cells.append(code("""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full heatmap
    im = axes[0].imshow(prices_raw, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[0].set_xlabel("Price dimension (224)")
    axes[0].set_ylabel("Day index")
    axes[0].set_title("Raw Swaption Surfaces (494 days)")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # Time series of a few representative dimensions
    dims = [0, 55, 111, 167, 223]
    for d in dims:
        axes[1].plot(prices_raw[:, d], alpha=0.7, linewidth=0.8, label=f"dim {d}")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Price")
    axes[1].set_title("Selected Price Dimensions Over Time")
    axes[1].legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.show()
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════
cells.append(md("""
    ---
    <a id="3-preprocessing"></a>
    ## 3 · Preprocessing Pipeline

    Raw swaption prices have **varying scales, outliers, and regime shifts**.
    We apply a three-stage normalization:

    1. **Winsorize** — clip to 1st / 99th percentile (dimension-wise)
    2. **Robust Scaling** — center by median, scale by IQR (handles outliers better than z-score)
    3. **MinMax Scaling** — rescale to \\[0, 1\\] for the AE's sigmoid output

    All scaler parameters are **fitted on training data only** — test data uses
    `transform()` to avoid leakage.
"""))

cells.append(code("""
    preprocessor = SwaptionPreprocessor(
        winsorize_limits=tuple(cfg["preprocessing"]["winsorize_limits"])
    )
    prices_norm = preprocessor.fit_transform(prices_raw)

    print(f"Normalised range : [{prices_norm.min():.4f}, {prices_norm.max():.4f}]")
    print(f"Mean             : {prices_norm.mean():.4f}")
    print(f"Std              : {prices_norm.std():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(prices_raw.ravel(), bins=100, alpha=0.7, color="#6366f1", edgecolor="white")
    axes[0].set_title("Before Preprocessing")
    axes[0].set_xlabel("Price")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(prices_norm.ravel(), bins=100, alpha=0.7, color="#8b5cf6", edgecolor="white")
    axes[1].set_title("After Preprocessing (normalised to [0,1])")
    axes[1].set_xlabel("Normalised price")

    plt.tight_layout()
    plt.show()
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 4 — AUTOENCODER
# ═══════════════════════════════════════════════════════════════
cells.append(md("""
    ---
    <a id="4-autoencoder"></a>
    ## 4 · Sparse Denoising Autoencoder

    We compress each 224-dim surface to a **20-dim latent code** using a symmetric
    autoencoder with:

    - **Denoising**: 15% random masking during training (Dropout-style)
    - **Sparsity**: L1 penalty (λ = 10⁻⁴) on the bottleneck to encourage compact representations
    - **Architecture**: `224 → 128 → 64 → 20 → 64 → 128 → 224`

    | Parameter | Value |
    |-----------|-------|
    | Input dim | 224 |
    | Latent dim | 20 |
    | Hidden layers | [128, 64] |
    | Mask ratio | 0.15 |
    | L1 sparsity λ | 10⁻⁴ |
    | Learning rate | 10⁻³ |
    | Val split | last 50 days |
"""))

cells.append(code("""
    # Load pre-trained autoencoder
    ae_cfg = cfg["autoencoder"]
    ae_path = os.path.join(PROJECT_ROOT, cfg["data"]["output_dir"], "ae_weights.pt")

    ae_model = load_autoencoder(
        ae_path,
        input_dim=ae_cfg["input_dim"],
        hidden_dims=tuple(ae_cfg["hidden_dims"]),
        latent_dim=ae_cfg["latent_dim"],
        device="cpu",
    )
    ae_model.eval()
    for p in ae_model.parameters():
        p.requires_grad_(False)

    n_params_ae = sum(p.numel() for p in ae_model.parameters())
    print(f"AE parameters : {n_params_ae:,}")
    print(f"Compression   : 224 → 20  ({224/20:.1f}× reduction)")
"""))

cells.append(md("""
    ### 4.1 · Reconstruction Quality

    We visualise how well the AE reconstructs the original surfaces
    by decoding the latent codes back to 224-dim space.
"""))

cells.append(code("""
    with torch.no_grad():
        x_tensor = torch.tensor(prices_norm, dtype=torch.float32)
        x_hat, z = ae_model(x_tensor)  # reconstruct
        latent_codes = z.numpy()
        reconstructed = x_hat.numpy()

    recon_mse = np.mean((prices_norm - reconstructed) ** 2)
    recon_r2 = 1 - np.sum((prices_norm - reconstructed)**2) / np.sum((prices_norm - prices_norm.mean(0))**2)

    print(f"Reconstruction MSE : {recon_mse:.6f}")
    print(f"Reconstruction R²  : {recon_r2:.4f}")
    print(f"Latent codes shape : {latent_codes.shape}")

    # Pick 4 random days to show reconstruction
    np.random.seed(42)
    sample_days = sorted(np.random.choice(prices_norm.shape[0], 4, replace=False))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, day in zip(axes.ravel(), sample_days):
        ax.plot(prices_norm[day], linewidth=1, alpha=0.8, label="Original", color="#374151")
        ax.plot(reconstructed[day], linewidth=1, alpha=0.8, label="Reconstructed", color="#8b5cf6", linestyle="--")
        ax.set_title(f"Day {day}", fontsize=10)
        ax.set_xlabel("Price dimension")
        ax.legend(fontsize=8)
    plt.suptitle("AE Reconstruction: Original vs Decoded", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 5 — LATENT SPACE
# ═══════════════════════════════════════════════════════════════
cells.append(md("""
    ---
    <a id="5-latent"></a>
    ## 5 · Latent Space Analysis

    The 20-dim latent codes capture the essential dynamics of the swaption market.
    Below we visualise:
    - **Temporal trajectories** of the top latent dimensions
    - **Correlation structure** between latent dimensions
"""))

cells.append(code("""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Top 6 latent dimensions by variance
    variances = latent_codes.var(axis=0)
    top_dims = np.argsort(variances)[::-1][:6]

    for d in top_dims:
        axes[0].plot(latent_codes[:, d], linewidth=0.8, alpha=0.8, label=f"z[{d}] (var={variances[d]:.3f})")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Latent value")
    axes[0].set_title("Temporal Dynamics of Top-Variance Latent Dimensions")
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")

    # Correlation matrix
    corr = np.corrcoef(latent_codes.T)
    im = axes[1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_xlabel("Latent dim")
    axes[1].set_ylabel("Latent dim")
    axes[1].set_title("Latent Dimension Correlation Matrix")
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.show()
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 6 — QUANTUM RESERVOIR
# ═══════════════════════════════════════════════════════════════
cells.append(md("""
    ---
    <a id="6-quantum"></a>
    ## 6 · Photonic Quantum Reservoir Computing

    The core innovation of our approach is the **Ensemble Quantum Optical
    Reservoir Computer (EnsembleQORC)** — three fixed photonic circuits that map
    classical context windows to a high-dimensional Fock-state feature space.

    ### Why photonic QRC?

    | Property | Benefit |
    |----------|---------|
    | **Fock-space features** | Exponentially many basis states from only a few modes/photons |
    | **Multi-photon interference** | Encodes permanents — #P-hard to simulate classically |
    | **No training needed** | Fixed random unitary "reservoir" — only the readout is trained |
    | **Ensemble diversity** | Three circuits with different mode/photon configs capture complementary features |

    ### Ensemble Configuration

    | Reservoir | Modes | Photons | Fock dim | Seed |
    |-----------|-------|---------|----------|------|
    | 1 | 12 | 3 | C(14,3) = 364 | 42 |
    | 2 | 10 | 4 | C(13,4) = 715 | 43 |
    | 3 | 16 | 2 | C(17,2) = 136 | 44 |
    | **Total** | — | — | **1,215** | — |

    Each circuit follows the **"sandwich" architecture**:

    ```
    Input → Linear(120 → n_modes) → Sigmoid → [0,1] angles
      → Haar-random Unitary → Phase Shifters (input-driven) → Same Unitary
      → Fock-state probability measurement
    ```
"""))

cells.append(code("""
    # Show the quantum feature dimensions
    from math import comb

    configs = [
        {"n_modes": 12, "n_photons": 3, "seed": 42},
        {"n_modes": 10, "n_photons": 4, "seed": 43},
        {"n_modes": 16, "n_photons": 2, "seed": 44},
    ]

    total_dim = 0
    for i, c in enumerate(configs):
        dim = comb(c["n_modes"] + c["n_photons"] - 1, c["n_photons"])
        total_dim += dim
        print(f"  Reservoir {i+1}: {c['n_modes']} modes, {c['n_photons']} photons "
              f"→ C({c['n_modes']+c['n_photons']-1},{c['n_photons']}) = {dim} features")

    print(f"  ─────────────────────────────────────")
    print(f"  Total quantum feature dimension: {total_dim}")
"""))

cells.append(md("""
    ### 6.1 · Pre-computed Quantum Features

    Quantum features are pre-computed once and saved. Let's load them and
    visualise their distribution.
"""))

cells.append(code("""
    # Load pre-computed quantum features
    Q_all = np.load(os.path.join(PROJECT_ROOT, "outputs/quantum_features.npy"))
    print(f"Quantum features shape: {Q_all.shape}")
    print(f"Range: [{Q_all.min():.4f}, {Q_all.max():.4f}]")
    print(f"Mean:  {Q_all.mean():.4f}")
    print(f"Std:   {Q_all.std():.4f}")
    print(f"Sparsity (fraction near 0): {(np.abs(Q_all) < 0.01).mean():.1%}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Distribution of all quantum features
    axes[0].hist(Q_all.ravel(), bins=100, alpha=0.7, color="#7c3aed", edgecolor="white")
    axes[0].set_title("Quantum Feature Distribution")
    axes[0].set_xlabel("Feature value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_yscale("log")

    # Feature variance across the 1215 dimensions
    q_var = Q_all.var(axis=0)
    axes[1].plot(sorted(q_var, reverse=True), color="#7c3aed", linewidth=0.8)
    axes[1].set_title("Quantum Feature Variance (sorted)")
    axes[1].set_xlabel("Feature rank")
    axes[1].set_ylabel("Variance")

    # Heatmap of first 50 features over time
    im = axes[2].imshow(Q_all[:, :50].T, aspect="auto", cmap="magma")
    axes[2].set_xlabel("Window index")
    axes[2].set_ylabel("Feature index")
    axes[2].set_title("Quantum Features (first 50 dims)")
    plt.colorbar(im, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.show()
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 7 — HYBRID MODEL
# ═══════════════════════════════════════════════════════════════
cells.append(md("""
    ---
    <a id="7-model"></a>
    ## 7 · Hybrid Model & Ridge Readout

    ### Temporal Windowing

    We create sliding windows of **5 consecutive latent codes** plus a
    **delta (first difference)** as context for predicting the next day:

    ```
    X = [z_{t-5}, z_{t-4}, z_{t-3}, z_{t-2}, z_{t-1}, Δz_{t-1}]  →  120 dims
    y = z_t                                                          →   20 dims
    ```

    ### Feature Concatenation & Ridge Readout

    Our **primary model (QORC + Ridge)** concatenates:
    - **Quantum features** (1,215 Fock-state probabilities from EnsembleQORC)
    - **Classical context** (120-dim windowed latent codes)

    → **1,335 total features** fed to a simple **Ridge regression** (α = 100).

    Ridge was chosen deliberately — it isolates the quantum reservoir's
    contribution. If a simple linear readout can achieve top accuracy, the
    expressiveness must reside in the feature space rather than the model.
"""))

cells.append(code("""
    # Create train/val splits
    window_size = cfg["hybrid_model"]["window_size"]
    val_split = cfg["autoencoder"]["val_split"]

    X_all, y_all, indices = make_windows(latent_codes, window_size=window_size)
    n_total = len(X_all)
    n_train = n_total - val_split

    X_train, X_val = X_all[:n_train], X_all[n_train:]
    y_train, y_val = y_all[:n_train], y_all[n_train:]

    # Quantum features
    Q_train, Q_val = Q_all[:n_train], Q_all[n_train:]

    print(f"Window size     : {window_size}")
    print(f"Total windows   : {n_total}")
    print(f"Train / Val     : {n_train} / {val_split}")
    print(f"Classical dim   : {X_all.shape[1]}")
    print(f"Quantum dim     : {Q_all.shape[1]}")
    print(f"Combined dim    : {X_all.shape[1] + Q_all.shape[1]}")
"""))

cells.append(md("""
    ### 7.1 · Training the Ridge Readout
"""))

cells.append(code("""
    from sklearn.linear_model import Ridge

    # Concatenate quantum + classical features
    X_tr_full = np.hstack([Q_train, X_train])
    X_vl_full = np.hstack([Q_val, X_val])

    # Grid search over alpha
    best_alpha, best_mse = None, float("inf")
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_tr_full, y_train)
        preds = model.predict(X_vl_full)
        mse_val = np.mean((preds - y_val) ** 2)
        print(f"  alpha={alpha:<6}  val MSE = {mse_val:.6f}")
        if mse_val < best_mse:
            best_mse, best_alpha = mse_val, alpha

    # Final model with best alpha
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_tr_full, y_train)
    val_preds = ridge.predict(X_vl_full)

    val_mse = np.mean((val_preds - y_val) ** 2)
    val_r2 = 1 - np.sum((val_preds - y_val)**2) / np.sum((y_val - y_val.mean(0))**2)

    print(f"\\n  Best alpha    : {best_alpha}")
    print(f"  Val MSE       : {val_mse:.6f}")
    print(f"  Val R²        : {val_r2:.4f}")
    print(f"  Ridge params  : {np.prod(ridge.coef_.shape) + np.prod(np.array(ridge.intercept_).shape):,}")
"""))

cells.append(md("""
    ### 7.2 · Validation — Predicted vs True Latent Codes
"""))

cells.append(code("""
    # Decode predictions and truth to surface level
    with torch.no_grad():
        surf_pred = ae_model.decode(torch.tensor(val_preds, dtype=torch.float32)).numpy()
        surf_true = ae_model.decode(torch.tensor(y_val, dtype=torch.float32)).numpy()

    surf_mse = np.mean((surf_pred - surf_true) ** 2)
    surf_rmse = np.sqrt(surf_mse)
    print(f"Surface MSE  : {surf_mse:.6f}")
    print(f"Surface RMSE : {surf_rmse:.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Latent: first 4 dims
    for i in range(4):
        ax = axes[i // 2, i % 2]
        ax.plot(y_val[:, i], label="True", color="#374151", linewidth=1)
        ax.plot(val_preds[:, i], label="Predicted", color="#8b5cf6", linewidth=1, linestyle="--")
        ax.set_title(f"Latent dim {i}", fontsize=10)
        ax.set_xlabel("Val sample")
        ax.legend(fontsize=8)

    plt.suptitle("QORC + Ridge: Predicted vs True Latent Codes (Validation)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 8 — TEST PREDICTIONS
# ═══════════════════════════════════════════════════════════════
cells.append(md("""
    ---
    <a id="8-test"></a>
    ## 8 · Test Predictions (6 Future Days)

    `test.xlsx` contains **6 actual future swaption surfaces** (days 495–500)
    that were never seen during training. We use **walk-forward evaluation**:

    - For each test day, the context window uses only data observed up to that point
    - No test data is used for training or fitting — only for comparison
    - Previous test-day actuals are used as context for subsequent predictions (walk-forward)
"""))

cells.append(code("""
    # Load test data
    _, _, test_prices_raw = load_train_data(
        os.path.join(PROJECT_ROOT, "DATASETS/test.xlsx")
    )
    print(f"Test rows: {test_prices_raw.shape[0]} × {test_prices_raw.shape[1]}")

    # Transform with FITTED preprocessor (no re-fitting!)
    test_prices_norm = preprocessor.transform(test_prices_raw)

    # Encode through frozen AE
    with torch.no_grad():
        test_latent = ae_model.encode(
            torch.tensor(test_prices_norm, dtype=torch.float32)
        ).numpy()

    print(f"Test latent shape: {test_latent.shape}")
    print(f"Test norm range  : [{test_prices_norm.min():.4f}, {test_prices_norm.max():.4f}]")
"""))

cells.append(code("""
    # Walk-forward windows: concatenate train + test latent codes
    all_latent = np.concatenate([latent_codes, test_latent], axis=0)
    X_all_ext, y_all_ext, _ = make_windows(all_latent, window_size=window_size)

    n_test = test_prices_raw.shape[0]
    X_test = X_all_ext[-n_test:]
    y_test = y_all_ext[-n_test:]

    # Compute quantum features for test windows
    set_seed(cfg["seed"])
    ensemble_configs = cfg.get("quantum_reservoir", {}).get("ensemble", [
        {"n_modes": 12, "n_photons": 3, "seed": 42},
        {"n_modes": 10, "n_photons": 4, "seed": 43},
        {"n_modes": 16, "n_photons": 2, "seed": 44},
    ])

    ensemble = EnsembleQORC(
        input_dim=X_test.shape[1],
        configs=ensemble_configs,
        use_fock=cfg.get("quantum_reservoir", {}).get("use_fock", True),
        device="cpu",
    )
    ensemble.eval()

    Q_test_raw = extract_quantum_features(
        ensemble, torch.tensor(X_test, dtype=torch.float32), batch_size=64
    )

    # Normalise with training stats
    X_train_val_t = torch.tensor(
        np.concatenate([X_train, X_val], axis=0), dtype=torch.float32
    )
    Q_train_raw = extract_quantum_features(ensemble, X_train_val_t, batch_size=64)

    normalizer = QuantumFeatureNormalizer()
    _ = normalizer.fit_transform(Q_train_raw)
    Q_test = normalizer.transform(Q_test_raw)

    print(f"Test quantum features: {Q_test.shape}")
"""))

cells.append(code("""
    # Predict with our trained Ridge model
    X_test_full = np.hstack([Q_test, X_test])
    test_preds = ridge.predict(X_test_full)

    test_mse = np.mean((test_preds - y_test) ** 2)
    test_r2 = 1 - np.sum((test_preds - y_test)**2) / np.sum((y_test - y_test.mean(0))**2)

    # Surface-level
    with torch.no_grad():
        test_surf_pred = ae_model.decode(torch.tensor(test_preds, dtype=torch.float32)).numpy()
        test_surf_true = ae_model.decode(torch.tensor(y_test, dtype=torch.float32)).numpy()

    test_surf_rmse = np.sqrt(np.mean((test_surf_pred - test_surf_true) ** 2))

    print(f"TEST RESULTS (QORC + Ridge)")
    print(f"  Latent MSE     : {test_mse:.6f}")
    print(f"  Latent R²      : {test_r2:.4f}")
    print(f"  Surface RMSE   : {test_surf_rmse:.6f}")
"""))

cells.append(md("""
    ### 8.1 · Test — Predicted vs Actual Surfaces
"""))

cells.append(code("""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for i, ax in enumerate(axes.ravel()):
        if i >= n_test:
            ax.set_visible(False)
            continue
        ax.plot(test_surf_true[i], linewidth=1, alpha=0.8, label="Actual", color="#374151")
        ax.plot(test_surf_pred[i], linewidth=1, alpha=0.8, label="Predicted", color="#8b5cf6", linestyle="--")
        day_err = np.sqrt(np.mean((test_surf_true[i] - test_surf_pred[i])**2))
        ax.set_title(f"Test Day {i+1}  (RMSE={day_err:.4f})", fontsize=10)
        ax.set_xlabel("Price dim")
        ax.legend(fontsize=8)

    plt.suptitle("QORC + Ridge: Predicted vs Actual Swaption Surfaces (6 Test Days)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 9 — BENCHMARKS
# ═══════════════════════════════════════════════════════════════
cells.append(md("""
    ---
    <a id="9-benchmarks"></a>
    ## 9 · Benchmark Comparison (11 Models)

    We benchmarked our approach against **10 competing methods** spanning
    classical ML, deep learning, and alternative quantum approaches.

    All models share the **same data, same splits, same AE**. They differ only
    in how the latent-code prediction is made.

    | # | Model | Type |
    |---|-------|------|
    | 1 | **★ QORC + Ridge (ours)** | Hybrid Quantum — Photonic QRC + Ridge |
    | 2 | **QUANTECH MLP (ours)** | Hybrid Quantum — Photonic QRC + MLP |
    | 3 | Classical LSTM | Deep Learning — 2×LSTM(64) |
    | 4 | Quantum LSTM | Hybrid — VQC-enhanced LSTM |
    | 5 | Random Forest | Ensemble Trees — 300 trees |
    | 6 | Ridge Regression | Linear — L2 regularised |
    | 7 | Gradient Boosting | Ensemble Trees — per-output |
    | 8 | SVR (RBF) | Kernel Method |
    | 9 | sklearn MLP | Neural Network |
    | 10 | VQC (Trained) | Variational Quantum Circuit |
    | 11 | Simple PML + Ridge | Single photonic layer + Ridge |
"""))

cells.append(code("""
    # Load benchmark results
    results = pd.read_csv("benchmarks/results.csv")

    # Display sorted by test latent MSE
    display_cols = [
        "name", "latent_MSE", "R2",
        "test_latent_MSE", "test_R2",
        "test_surface_RMSE", "train_time", "inference_ms"
    ]
    results_sorted = results.sort_values("test_latent_MSE").reset_index(drop=True)
    results_sorted.index += 1
    results_sorted.index.name = "Rank"
    results_sorted[display_cols].round(6)
"""))

cells.append(md("""
    ### 9.1 · Benchmark Figures

    Below we generate the key comparison figures from the benchmark results.
"""))

cells.append(code("""
    # ── Colour palette ──
    QUANTUM_NAMES = {
        "Simple PML + Ridge", "★ QORC + Ridge (ours)",
        "QUANTECH MLP (ours)", "Quantum LSTM", "VQC (Trained)",
    }
    VIOLET = "#7c3aed"
    VIOLET_L = "#c4b5fd"
    EMERALD = "#059669"
    EMERALD_L = "#a7f3d0"
    OURS_C = "#8b5cf6"
    OURS_L = "#ddd6fe"

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
"""))

cells.append(md("""
    ### 9.2 · Test Latent MSE Comparison
"""))

cells.append(code("""
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        range(len(df)), df["test_latent_MSE"],
        color=df["fill"].values, edgecolor=df["edge"].values, linewidth=1.5
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["short"].values, fontsize=9)
    ax.set_xlabel("Test Latent MSE (lower is better)")
    ax.set_title("Benchmark: Test Latent MSE", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, df["test_latent_MSE"])):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)

    legend_patches = [
        Patch(facecolor=OURS_L, edgecolor=OURS_C, label="Ours (Quantum)"),
        Patch(facecolor=VIOLET_L, edgecolor=VIOLET, label="Other Quantum"),
        Patch(facecolor=EMERALD_L, edgecolor=EMERALD, label="Classical"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()
"""))

cells.append(md("""
    ### 9.3 · Test Surface RMSE Comparison
"""))

cells.append(code("""
    df_srmse = df.sort_values("test_surface_RMSE").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        range(len(df_srmse)), df_srmse["test_surface_RMSE"],
        color=df_srmse["fill"].values, edgecolor=df_srmse["edge"].values, linewidth=1.5
    )
    ax.set_yticks(range(len(df_srmse)))
    ax.set_yticklabels(df_srmse["short"].values, fontsize=9)
    ax.set_xlabel("Test Surface RMSE (lower is better)")
    ax.set_title("Benchmark: Test Surface RMSE (What Matters for Pricing)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars, df_srmse["test_surface_RMSE"])):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)

    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()
"""))

cells.append(md("""
    ### 9.4 · Test R² Score
"""))

cells.append(code("""
    df_r2 = df.sort_values("test_R2", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        range(len(df_r2)), df_r2["test_R2"],
        color=df_r2["fill"].values, edgecolor=df_r2["edge"].values, linewidth=1.5
    )
    ax.set_yticks(range(len(df_r2)))
    ax.set_yticklabels(df_r2["short"].values, fontsize=9)
    ax.set_xlabel("Test R² (higher is better)")
    ax.set_title("Benchmark: Test R²", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(0, color="#ef4444", linestyle="--", linewidth=0.8, alpha=0.5)

    for i, (bar, val) in enumerate(zip(bars, df_r2["test_R2"])):
        x_pos = max(bar.get_width(), 0) + 0.02
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)

    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()
"""))

cells.append(md("""
    ### 9.5 · Accuracy vs Inference Speed
"""))

cells.append(code("""
    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in df.iterrows():
        c = OURS_C if row["is_ours"] else (VIOLET if row["is_quantum"] else EMERALD)
        marker = "*" if row["is_ours"] else ("D" if row["is_quantum"] else "o")
        size = 200 if row["is_ours"] else 80

        ax.scatter(
            row["inference_ms"], row["test_latent_MSE"],
            color=c, s=size, marker=marker, edgecolors="white", linewidths=0.5,
            zorder=5
        )
        ax.annotate(
            row["short"], (row["inference_ms"], row["test_latent_MSE"]),
            fontsize=7, ha="left", va="bottom",
            xytext=(5, 3), textcoords="offset points"
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference Latency (ms, log scale)")
    ax.set_ylabel("Test Latent MSE (lower is better)")
    ax.set_title("Accuracy vs Speed Trade-off", fontsize=13, fontweight="bold")
    ax.legend(handles=legend_patches, fontsize=9)
    plt.tight_layout()
    plt.show()
"""))

cells.append(md("""
    ### 9.6 · Validation vs Test Performance

    Models that generalise well should appear **near the diagonal**.
    Overfitting models will have much worse test performance than validation.
"""))

cells.append(code("""
    fig, ax = plt.subplots(figsize=(8, 8))

    for _, row in df.iterrows():
        c = OURS_C if row["is_ours"] else (VIOLET if row["is_quantum"] else EMERALD)
        marker = "*" if row["is_ours"] else ("D" if row["is_quantum"] else "o")
        size = 200 if row["is_ours"] else 80

        ax.scatter(
            row["latent_MSE"], row["test_latent_MSE"],
            color=c, s=size, marker=marker, edgecolors="white", linewidths=0.5,
            zorder=5
        )
        ax.annotate(
            row["short"], (row["latent_MSE"], row["test_latent_MSE"]),
            fontsize=7, ha="left", va="bottom",
            xytext=(4, 3), textcoords="offset points"
        )

    # Diagonal line
    lims = [0, max(df["latent_MSE"].max(), df["test_latent_MSE"].max()) * 1.1]
    ax.plot(lims, lims, "--", color="#9ca3af", linewidth=1, alpha=0.7, label="Perfect generalisation")

    ax.set_xlabel("Validation Latent MSE")
    ax.set_ylabel("Test Latent MSE")
    ax.set_title("Validation vs Test: Generalisation Check", fontsize=13, fontweight="bold")
    ax.legend(handles=legend_patches + [plt.Line2D([0],[0], linestyle="--", color="#9ca3af", label="y=x")],
              fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
"""))

cells.append(md("""
    ### 9.7 · Training Time
"""))

cells.append(code("""
    df_time = df.sort_values("train_time", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        range(len(df_time)), df_time["train_time"],
        color=df_time["fill"].values, edgecolor=df_time["edge"].values, linewidth=1.5
    )
    ax.set_yticks(range(len(df_time)))
    ax.set_yticklabels(df_time["short"].values, fontsize=9)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Training Time Comparison", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars, df_time["train_time"])):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}s", va="center", fontsize=8)

    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 10 — CONCLUSION
# ═══════════════════════════════════════════════════════════════
cells.append(md("""
    ---
    <a id="10-conclusion"></a>
    ## 10 · Conclusion & Quantum Advantage

    ### Key Results

    | Metric | QORC + Ridge | Best Classical | Improvement |
    |--------|-------------|---------------|-------------|
    | **Test Surface RMSE** | **0.0425** | 0.0445 (LSTM) | 4.5% better |
    | **Test Latent MSE** | 0.0099 | 0.0091 (Ridge) | close |
    | **Inference** | 0.10 ms | 0.06 ms (Ridge) | comparable |
    | **Training** | 0.23 s | 0.01 s (Ridge) | fast |

    ### Why Photonic QRC Works

    1. **Exponential feature space** — 1,215 Fock-state features from just 3 small
       circuits (12/10/16 modes). Classical random features of the same dimension
       would lack the structured non-linearity from multi-photon interference.

    2. **No barren plateaus** — Unlike VQC/QAOA, the reservoir is **fixed**. We
       only train a linear readout, avoiding the trainability problems that plague
       variational quantum algorithms.

    3. **Best surface-level accuracy** — Despite Ridge Regression having lower
       latent MSE, QORC + Ridge achieves the **best surface RMSE** (0.0425),
       which is what matters for actual swaption pricing. The quantum features
       capture non-linear surface structure that latent MSE alone doesn't reflect.

    4. **Hardware-ready** — The same circuits run natively on Quandela's photonic
       processors. On-chip inference would be **nanoseconds** at **microwatt** power,
       vs milliseconds at ~10W for classical simulation.

    5. **Ensemble diversity** — Three circuits with different mode/photon configurations
       provide complementary views of the data, analogous to random forests but with
       quantum correlations.

    ### Limitations & Future Work

    - Only 6 test days — more data needed for robust statistical comparison
    - Classical simulation of Fock probabilities is slow for large mode counts
    - Current approach is **reservoir computing** (fixed circuit) — a hybrid
      trainable + fixed approach might improve further
    - Surface-level evaluation depends entirely on the AE's reconstruction quality

    ---
    *QUANTECH — Q-volution Hackathon 2026*
"""))

# ═══════════════════════════════════════════════════════════════
# BUILD NOTEBOOK
# ═══════════════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantech.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written: {out_path}")
print(f"Total cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='markdown')} markdown + {sum(1 for c in cells if c['cell_type']=='code')} code)")
