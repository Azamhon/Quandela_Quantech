"""
QORC + Ridge Regression model for swaption surface prediction.

Core idea: Replace the MLP ClassicalHead with sklearn Ridge regression.
This is the textbook reservoir computing readout — a linear readout on
nonlinear reservoir features.

Uses the same QORC ensemble features (1215 dims) + classical context
(120 dims) = 1335 input features.
"""

import os
import time
import math
import numpy as np
import torch
import joblib

from sklearn.linear_model import Ridge

# ── Project imports ───────────────────────────────────────────
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.quantum_reservoir import (
    EnsembleQORC, extract_quantum_features, QuantumFeatureNormalizer
)


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train_ridge(X_context, Q_features, y_latent, val_split,
                alpha=1.0, ae_model=None, verbose=True):
    """
    Train a Ridge regression on QORC features + classical context.

    Args:
        X_context   : np.ndarray (N, 120) — windowed latent context
        Q_features  : np.ndarray (N, 1215) — pre-computed quantum features
        y_latent    : np.ndarray (N, 20)  — target latent codes
        val_split   : int — number of last samples for validation
        alpha       : float — Ridge regularisation strength
        ae_model    : optional AE for surface-level reporting
        verbose     : bool

    Returns:
        ridge_model : fitted sklearn Ridge
        results     : dict with metrics
    """
    # Concatenate quantum + classical features
    X_full = np.hstack([Q_features, X_context])  # (N, 1335)
    n_total = len(X_full)
    n_train = n_total - val_split

    X_train, X_val = X_full[:n_train], X_full[n_train:]
    y_train, y_val = y_latent[:n_train], y_latent[n_train:]

    # Fit Ridge
    t0 = time.perf_counter()
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # Predictions
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    # Latent metrics
    train_mse = float(np.mean((pred_train - y_train) ** 2))
    val_mse = float(np.mean((pred_val - y_val) ** 2))
    train_rmse = math.sqrt(train_mse)
    val_rmse = math.sqrt(val_mse)

    # R²
    ss_res = np.sum((pred_val - y_val) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val, axis=0)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-10))

    # Surface metrics (if AE available)
    surface_mse = None
    surface_rmse = None
    if ae_model is not None:
        with torch.no_grad():
            pred_surf = ae_model.decode(
                torch.tensor(pred_val, dtype=torch.float32)
            ).numpy()
            true_surf = ae_model.decode(
                torch.tensor(y_val, dtype=torch.float32)
            ).numpy()
        surface_mse = float(np.mean((pred_surf - true_surf) ** 2))
        surface_rmse = math.sqrt(surface_mse)

    # Parameter count
    n_params = int(np.prod(model.coef_.shape))
    if model.intercept_ is not None:
        n_params += int(np.prod(np.array(model.intercept_).shape))

    if verbose:
        print(f"  Ridge alpha={alpha}")
        print(f"    Input dim        : {X_full.shape[1]}")
        print(f"    Train MSE        : {train_mse:.6f} (RMSE={train_rmse:.6f})")
        print(f"    Val   MSE        : {val_mse:.6f} (RMSE={val_rmse:.6f})")
        print(f"    Val   R²         : {r2:.4f}")
        if surface_mse is not None:
            print(f"    Surface MSE      : {surface_mse:.6f} (RMSE={surface_rmse:.6f})")
        print(f"    Parameters       : {n_params:,}")
        print(f"    Train time       : {train_time:.3f}s")

    results = {
        "model": model,
        "predictions": pred_val,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "r2": r2,
        "surface_mse": surface_mse,
        "surface_rmse": surface_rmse,
        "n_parameters": n_params,
        "train_time": train_time,
        "alpha": alpha,
    }
    return model, results


def search_alpha(X_context, Q_features, y_latent, val_split,
                 alphas=None, ae_model=None, verbose=True):
    """
    Try multiple alpha values and return the best model.

    Args:
        alphas: list of float — alpha values to try
    Returns:
        best_model, best_results, all_results
    """
    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    if verbose:
        print(f"\n  Searching {len(alphas)} alpha values: {alphas}")

    best_model = None
    best_results = None
    best_mse = float("inf")
    all_results = {}

    for alpha in alphas:
        model, results = train_ridge(
            X_context, Q_features, y_latent, val_split,
            alpha=alpha, ae_model=ae_model, verbose=verbose,
        )
        all_results[alpha] = results
        if results["val_mse"] < best_mse:
            best_mse = results["val_mse"]
            best_model = model
            best_results = results

    if verbose:
        print(f"\n  Best alpha = {best_results['alpha']} "
              f"(val MSE = {best_mse:.6f})")

    return best_model, best_results, all_results


# ─────────────────────────────────────────────────────────────
# Single-step inference (for autoregressive rollout)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_next_latent_ridge(context_vec, ridge_model, ensemble, qf_norm, device):
    """
    Predict the next latent code using Ridge model.

    Args:
        context_vec : np.ndarray (classical_dim,) — windowed latent context
        ridge_model : fitted sklearn Ridge
        ensemble    : EnsembleQORC
        qf_norm     : QuantumFeatureNormalizer
        device      : torch.device

    Returns:
        np.ndarray (latent_dim,)
    """
    ctx_t = torch.tensor(context_vec, dtype=torch.float32).unsqueeze(0).to(device)

    # Quantum features
    q_raw = extract_quantum_features(ensemble, ctx_t, batch_size=1)  # (1, Q_dim)
    q_n = qf_norm.transform(q_raw)  # normalised

    # Concatenate and predict
    x_full = np.hstack([q_n, context_vec.reshape(1, -1)])  # (1, 1335)
    z_pred = ridge_model.predict(x_full)  # (1, latent_dim)
    return z_pred[0]


# ─────────────────────────────────────────────────────────────
# Task A: Future prediction (autoregressive rollout)
# ─────────────────────────────────────────────────────────────

def build_context(latent_window):
    """
    Build context vector from a window of latent codes.
    latent_window: np.ndarray (window_size, latent_dim)
    Returns: context vector (window_size*latent_dim + latent_dim,)
    """
    flat = latent_window.reshape(-1)
    delta = latent_window[-1] - latent_window[-2]
    return np.concatenate([flat, delta]).astype(np.float32)


@torch.no_grad()
def predict_future_ridge(
    n_steps, latent_codes, ridge_model, ensemble, qf_norm,
    ae_model, preprocessor, window_size, device
):
    """
    Autoregressively predict n_steps future surfaces using Ridge model.

    Args:
        n_steps      : int — number of future steps
        latent_codes : np.ndarray (N, latent_dim)
        ridge_model  : fitted sklearn Ridge
        ensemble     : EnsembleQORC
        qf_norm      : QuantumFeatureNormalizer
        ae_model     : frozen AE
        preprocessor : SwaptionPreprocessor
        window_size  : int
        device       : torch.device

    Returns:
        predictions: list of np.ndarray (224,) — raw price scale
    """
    window = latent_codes[-window_size:].copy()
    predictions = []

    for step in range(n_steps):
        ctx = build_context(window)
        z_next = predict_next_latent_ridge(ctx, ridge_model, ensemble, qf_norm, device)

        # Decode to normalised surface
        z_t = torch.tensor(z_next, dtype=torch.float32).unsqueeze(0).to(device)
        surf_n = ae_model.decode(z_t).cpu().numpy()[0]

        # Reverse normalisation → actual prices
        prices = preprocessor.inverse_transform(surf_n.reshape(1, -1))[0]
        predictions.append(prices)

        # Slide window forward
        window = np.vstack([window[1:], z_next])

    return predictions


# ─────────────────────────────────────────────────────────────
# Task B: Missing data imputation (reuses AE — same as predict.py)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def impute_missing_ridge(
    partial_prices, preprocessor, ae_model, device,
    train_prices=None, row_date=None
):
    """
    Impute missing values using the denoising AE (shared with MLP model).
    This is identical to the imputation in predict.py since both models
    share the same AE.
    """
    nan_mask = np.isnan(partial_prices)
    filled = partial_prices.copy()

    if train_prices is not None:
        obs_mask = ~nan_mask
        obs_vals = partial_prices[obs_mask]
        dists = np.sqrt(((train_prices[:, obs_mask] - obs_vals) ** 2).mean(axis=1))
        nearest_idx = int(np.argmin(dists))
        filled[nan_mask] = train_prices[nearest_idx, nan_mask]
    else:
        filled[nan_mask] = preprocessor.median_[nan_mask]

    norm = preprocessor.transform(filled.reshape(1, -1))[0]
    norm_t = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(device)
    z = ae_model.encode(norm_t)
    recon = ae_model.decode(z).cpu().numpy()[0]

    recon_prices = preprocessor.inverse_transform(recon.reshape(1, -1))[0]

    full_prices = partial_prices.copy()
    full_prices[nan_mask] = recon_prices[nan_mask]
    return full_prices


# ─────────────────────────────────────────────────────────────
# Save / Load helpers
# ─────────────────────────────────────────────────────────────

def save_ridge_model(model, path):
    """Save Ridge model using joblib."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    joblib.dump(model, path)
    print(f"  Saved Ridge model → {path}")


def load_ridge_model(path):
    """Load Ridge model from joblib."""
    model = joblib.load(path)
    print(f"  Loaded Ridge model ← {path}")
    return model
