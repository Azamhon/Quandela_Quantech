"""
Prediction pipeline — generates the submission file.

Handles both test row types:
  "Future prediction" — all 224 price columns are NaN.
                        We autoregressively roll the QRC forward from
                        the last known latent codes.

  "Missing data"      — a small subset of columns are NaN (4-6 values).
                        We pass the partial observation through the
                        denoising AE; it reconstructs the full surface.

Output:
    outputs/predictions.xlsx  — matching the sample output format exactly.

Usage:
    python src/predict.py
    python src/predict.py --config configs/config.yaml
"""

import argparse
import os
import sys

import numpy as np
import torch
from openpyxl import load_workbook
from openpyxl import Workbook

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils           import load_config, set_seed, get_device
from src.preprocessing   import (load_train_data, load_test_data,
                                  SwaptionPreprocessor)
from src.autoencoder     import load_autoencoder
from src.hybrid_model    import ClassicalHead, make_windows
from src.quantum_reservoir import (EnsembleQORC, extract_quantum_features,
                                    QuantumFeatureNormalizer)


# ─────────────────────────────────────────────────────────────
# Load artefacts produced by train.py
# ─────────────────────────────────────────────────────────────

def load_artefacts(cfg, device):
    """Load all saved checkpoints + fitted scalers."""
    out = cfg["data"]["output_dir"]
    ae_cfg   = cfg["autoencoder"]
    qrc_cfg  = cfg["quantum_reservoir"]
    hcfg     = cfg["hybrid_model"]

    # ── Preprocessor ──────────────────────────────────────────
    prep_data = np.load(os.path.join(out, "preprocessor.npz"))
    preprocessor = SwaptionPreprocessor()
    preprocessor.median_     = prep_data["median"]
    preprocessor.iqr_        = prep_data["iqr"]
    preprocessor.min_        = prep_data["min"]
    preprocessor.range_      = prep_data["range"]
    preprocessor.clip_lower_ = prep_data["clip_lower"]
    preprocessor.clip_upper_ = prep_data["clip_upper"]
    preprocessor.is_fitted = True

    # ── AE ────────────────────────────────────────────────────
    ae_model = load_autoencoder(
        os.path.join(out, "ae_weights.pt"),
        input_dim   = ae_cfg["input_dim"],
        hidden_dims = tuple(ae_cfg["hidden_dims"]),
        latent_dim  = ae_cfg["latent_dim"],
        device      = str(device),
    )
    ae_model.eval()

    # ── Latent codes (training sequence) ─────────────────────
    latent_codes = np.load(os.path.join(out, "latent_codes.npy"))  # (494, latent_dim)

    # ── Quantum feature normalizer ────────────────────────────
    # Re-fit on training quantum features (cached)
    Q_train = np.load(os.path.join(out, "quantum_features.npy"))
    window_size = hcfg["window_size"]
    val_split   = ae_cfg["val_split"]
    n_train_windows = len(latent_codes) - window_size - val_split
    qf_norm = QuantumFeatureNormalizer()
    qf_norm.fit_transform(Q_train[:n_train_windows])   # re-fit to get mean/std

    # ── Ensemble QORC ─────────────────────────────────────────
    classical_dim = ae_cfg["latent_dim"] * (window_size + 1)
    ensemble = EnsembleQORC(
        input_dim = classical_dim,
        configs   = qrc_cfg["ensemble"],
        use_fock  = qrc_cfg["use_fock"],
        device    = str(device),
    ).to(device)

    # ── Classical head ────────────────────────────────────────
    head = ClassicalHead(
        quantum_dim   = ensemble.total_output_dim,
        classical_dim = classical_dim,
        latent_dim    = ae_cfg["latent_dim"],
        hidden_dims   = tuple(hcfg["hidden_dims"]),
        dropout       = 0.0,   # no dropout at inference
    ).to(device)
    head.load_state_dict(
        torch.load(os.path.join(out, "head_weights.pt"), map_location=device, weights_only=True)
    )
    head.eval()

    print("  All artefacts loaded.")
    return preprocessor, ae_model, latent_codes, qf_norm, ensemble, head


# ─────────────────────────────────────────────────────────────
# Context builder (for one prediction step)
# ─────────────────────────────────────────────────────────────

def build_context(latent_window):
    """
    latent_window: np.ndarray (window_size, latent_dim)
    Returns context vector (window_size*latent_dim + latent_dim,)
    """
    flat  = latent_window.reshape(-1)
    delta = latent_window[-1] - latent_window[-2]
    return np.concatenate([flat, delta]).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Single-step QRC inference
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_next_latent(context_vec, ensemble, qf_norm, head, device):
    """
    context_vec: np.ndarray (classical_dim,)
    Returns: np.ndarray (latent_dim,)
    """
    ctx_t = torch.tensor(context_vec, dtype=torch.float32).unsqueeze(0).to(device)

    # Quantum features
    q_raw = extract_quantum_features(ensemble, ctx_t, batch_size=1)   # (1, Q_dim)
    q_n   = qf_norm.transform(q_raw)                                  # normalised
    q_t   = torch.tensor(q_n, dtype=torch.float32).to(device)

    # Head → latent prediction
    z_pred = head(q_t, ctx_t)                                         # (1, latent_dim)
    return z_pred.cpu().numpy()[0]


# ─────────────────────────────────────────────────────────────
# Task A: Future prediction  (all columns NaN)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_future(
    n_steps, latent_codes, ensemble, qf_norm, head, ae_model,
    preprocessor, window_size, device
):
    """
    Autoregressively predict n_steps future surfaces.

    Starts from the last `window_size` known latent codes and rolls forward.

    Returns:
        predictions: list of np.ndarray (224,) — raw price scale
    """
    # Seed window: last `window_size` known latent codes
    window = latent_codes[-window_size:].copy()   # (window_size, latent_dim)
    predictions = []

    for step in range(n_steps):
        ctx  = build_context(window)
        z_next = predict_next_latent(ctx, ensemble, qf_norm, head, device)

        # Decode to normalised surface
        z_t   = torch.tensor(z_next, dtype=torch.float32).unsqueeze(0).to(device)
        surf_n = ae_model.decode(z_t).cpu().numpy()[0]   # (224,) in [0,1]

        # Reverse normalisation → actual prices
        prices = preprocessor.inverse_transform(surf_n.reshape(1, -1))[0]
        predictions.append(prices)

        # Slide window forward (autoregressive)
        window = np.vstack([window[1:], z_next])

    return predictions


# ─────────────────────────────────────────────────────────────
# Task B: Missing data imputation  (few columns NaN)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def impute_missing(
    partial_prices, preprocessor, ae_model, device,
    train_prices=None, row_date=None
):
    """
    Impute missing values in a partial swaption surface.

    Strategy:
      1. Fill NaN positions with values from the temporally nearest training
         row (if available), falling back to training column medians.
      2. Normalise the partially-filled surface.
      3. Pass through denoising AE (which was trained to reconstruct from
         masked inputs → naturally handles missing values).
      4. Inverse-transform to recover actual prices.
      5. Use AE-reconstructed values ONLY for the originally-NaN positions;
         keep observed values as-is.

    Args:
        partial_prices: np.ndarray (224,) with np.nan where data is missing
        train_prices:   np.ndarray (N, 224) raw training prices (optional)
        row_date:       date of this row for temporal neighbor lookup (optional)

    Returns:
        full_prices: np.ndarray (224,) all values filled
    """
    nan_mask = np.isnan(partial_prices)

    # Fill NaN — prefer temporally nearest training row, fall back to medians
    filled = partial_prices.copy()
    if train_prices is not None:
        # Use the observed (non-NaN) columns to find the closest training row
        obs_mask = ~nan_mask
        obs_vals = partial_prices[obs_mask]
        dists = np.sqrt(((train_prices[:, obs_mask] - obs_vals) ** 2).mean(axis=1))
        nearest_idx = int(np.argmin(dists))
        filled[nan_mask] = train_prices[nearest_idx, nan_mask]
    else:
        filled[nan_mask] = preprocessor.median_[nan_mask]

    # Normalise
    norm = preprocessor.transform(filled.reshape(1, -1))[0]  # (224,)

    # AE reconstruction (denoising AE reconstructs full surface)
    norm_t = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(device)
    z      = ae_model.encode(norm_t)
    recon  = ae_model.decode(z).cpu().numpy()[0]          # (224,) in [0,1]

    # Inverse transform reconstructed surface
    recon_prices = preprocessor.inverse_transform(recon.reshape(1, -1))[0]

    # Merge: keep observed values, fill only the missing ones from AE
    full_prices = partial_prices.copy()
    full_prices[nan_mask] = recon_prices[nan_mask]

    return full_prices


# ─────────────────────────────────────────────────────────────
# Output writer
# ─────────────────────────────────────────────────────────────

def write_predictions(test_info, predictions, price_columns, out_path):
    """
    Write predictions to xlsx matching the sample output format.

    Output column order: [224 price cols | Date | Type]
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Predictions"

    # Header row
    header = price_columns + ["Date", "Type"]
    ws.append(header)

    for info, pred_prices in zip(test_info, predictions):
        row = list(pred_prices) + [str(info["date"]), info["type"]]
        ws.append(row)

    wb.save(out_path)
    print(f"\n  Predictions saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args):
    cfg    = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg)

    print("\n" + "=" * 60)
    print("  Prediction Pipeline")
    print("=" * 60)
    print(f"\nDevice: {device}")

    # ── Load artefacts ─────────────────────────────────────────
    print("\n[1] Loading artefacts...")
    (preprocessor, ae_model, latent_codes,
     qf_norm, ensemble, head) = load_artefacts(cfg, device)

    window_size = cfg["hybrid_model"]["window_size"]

    # Load raw training prices for nearest-neighbor imputation
    _, _, train_prices_raw = load_train_data(cfg["data"]["train_path"])

    # ── Load test data ─────────────────────────────────────────
    print("\n[2] Loading test data...")
    test_info, price_columns = load_test_data(cfg["data"]["test_path"])
    n_future  = sum(1 for r in test_info if r["type"] == "Future prediction")
    n_missing = sum(1 for r in test_info if r["type"] == "Missing data")
    print(f"  Future prediction rows : {n_future}")
    print(f"  Missing data rows      : {n_missing}")

    # ── Run predictions ────────────────────────────────────────
    print("\n[3] Generating predictions...")
    all_predictions = []

    # Task A: future prediction (autoregressive)
    future_prices = predict_future(
        n_steps      = n_future,
        latent_codes = latent_codes,
        ensemble     = ensemble,
        qf_norm      = qf_norm,
        head         = head,
        ae_model     = ae_model,
        preprocessor = preprocessor,
        window_size  = window_size,
        device       = device,
    )

    # Task B: missing data imputation (uses nearest training row for fill)
    missing_prices = []
    for info in test_info:
        if info["type"] == "Missing data":
            full = impute_missing(
                info["values"], preprocessor, ae_model, device,
                train_prices=train_prices_raw, row_date=info["date"]
            )
            missing_prices.append(full)

    # Merge in original row order
    fi, mi = 0, 0
    for info in test_info:
        if info["type"] == "Future prediction":
            all_predictions.append(future_prices[fi])
            fi += 1
        else:
            all_predictions.append(missing_prices[mi])
            mi += 1

    # ── Print summary ──────────────────────────────────────────
    print("\n[4] Prediction summary:")
    for i, (info, pred) in enumerate(zip(test_info, all_predictions)):
        print(
            f"  Row {i+1} [{info['type'][:7]}] "
            f"date={info['date']}  "
            f"price range=[{pred.min():.4f}, {pred.max():.4f}]  "
            f"mean={pred.mean():.4f}"
        )

    # ── Write output ───────────────────────────────────────────
    out_path = os.path.join(cfg["data"]["output_dir"], "predictions.xlsx")
    write_predictions(test_info, all_predictions, price_columns, out_path)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config YAML"
    )
    args = parser.parse_args()
    main(args)
