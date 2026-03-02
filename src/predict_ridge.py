"""
Prediction pipeline for QORC + Ridge Regression model.

Generates outputs/predictions_ridge.xlsx for hackathon submission.
Handles both "Future prediction" and "Missing data" task types
identically to the existing predict.py.

Usage:
    python src/predict_ridge.py
    python src/predict_ridge.py --config configs/config.yaml
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_seed, get_device
from src.preprocessing import (
    load_train_data, load_test_data, SwaptionPreprocessor,
)
from src.autoencoder import load_autoencoder
from src.quantum_reservoir import (
    EnsembleQORC, extract_quantum_features, QuantumFeatureNormalizer,
)
from src.ridge_model import (
    load_ridge_model, predict_future_ridge, impute_missing_ridge,
)
from src.predict import write_predictions


# ─────────────────────────────────────────────────────────────
# Load artefacts
# ─────────────────────────────────────────────────────────────

def load_artefacts(cfg, device):
    """Load all saved checkpoints + fitted scalers for Ridge model."""
    out = cfg["data"]["output_dir"]
    ae_cfg = cfg["autoencoder"]
    qrc_cfg = cfg["quantum_reservoir"]
    hcfg = cfg["hybrid_model"]

    # ── Preprocessor ──────────────────────────────────────────
    prep_data = np.load(os.path.join(out, "preprocessor.npz"))
    preprocessor = SwaptionPreprocessor()
    preprocessor.median_ = prep_data["median"]
    preprocessor.iqr_ = prep_data["iqr"]
    preprocessor.min_ = prep_data["min"]
    preprocessor.range_ = prep_data["range"]
    preprocessor.clip_lower_ = prep_data["clip_lower"]
    preprocessor.clip_upper_ = prep_data["clip_upper"]
    preprocessor.is_fitted = True

    # ── AE ────────────────────────────────────────────────────
    ae_model = load_autoencoder(
        os.path.join(out, "ae_weights.pt"),
        input_dim=ae_cfg["input_dim"],
        hidden_dims=tuple(ae_cfg["hidden_dims"]),
        latent_dim=ae_cfg["latent_dim"],
        device=str(device),
    )
    ae_model.eval()

    # ── Latent codes (training sequence) ─────────────────────
    latent_codes = np.load(os.path.join(out, "latent_codes.npy"))

    # ── Quantum feature normalizer (re-fit on training features) ──
    Q_train = np.load(os.path.join(out, "quantum_features.npy"))
    window_size = hcfg["window_size"]
    val_split = ae_cfg["val_split"]
    n_train_windows = len(latent_codes) - window_size - val_split
    qf_norm = QuantumFeatureNormalizer()
    qf_norm.fit_transform(Q_train[:n_train_windows])

    # ── Ensemble QORC ─────────────────────────────────────────
    classical_dim = ae_cfg["latent_dim"] * (window_size + 1)
    ensemble = EnsembleQORC(
        input_dim=classical_dim,
        configs=qrc_cfg["ensemble"],
        use_fock=qrc_cfg["use_fock"],
        device=str(device),
    ).to(device)

    # ── Ridge model ───────────────────────────────────────────
    ridge_path = os.path.join(out, "ridge_model.joblib")
    ridge_model = load_ridge_model(ridge_path)

    print("  All artefacts loaded (Ridge).")
    return preprocessor, ae_model, latent_codes, qf_norm, ensemble, ridge_model


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args):
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg)

    print("\n" + "=" * 60)
    print("  Prediction Pipeline (QORC + Ridge)")
    print("=" * 60)
    print(f"\nDevice: {device}")

    # ── Load artefacts ─────────────────────────────────────────
    print("\n[1] Loading artefacts...")
    (preprocessor, ae_model, latent_codes,
     qf_norm, ensemble, ridge_model) = load_artefacts(cfg, device)

    window_size = cfg["hybrid_model"]["window_size"]

    # Load raw training prices for nearest-neighbor imputation
    _, _, train_prices_raw = load_train_data(cfg["data"]["train_path"])

    # ── Load test data ─────────────────────────────────────────
    print("\n[2] Loading test data...")
    test_info, price_columns = load_test_data(cfg["data"]["test_path"])
    n_future = sum(1 for r in test_info if r["type"] == "Future prediction")
    n_missing = sum(1 for r in test_info if r["type"] == "Missing data")
    print(f"  Future prediction rows : {n_future}")
    print(f"  Missing data rows      : {n_missing}")

    # ── Run predictions ────────────────────────────────────────
    print("\n[3] Generating predictions (Ridge)...")
    all_predictions = []

    # Task A: future prediction (autoregressive with Ridge)
    future_prices = predict_future_ridge(
        n_steps=n_future,
        latent_codes=latent_codes,
        ridge_model=ridge_model,
        ensemble=ensemble,
        qf_norm=qf_norm,
        ae_model=ae_model,
        preprocessor=preprocessor,
        window_size=window_size,
        device=device,
    )

    # Task B: missing data imputation (uses AE — shared)
    missing_prices = []
    for info in test_info:
        if info["type"] == "Missing data":
            full = impute_missing_ridge(
                info["values"], preprocessor, ae_model, device,
                train_prices=train_prices_raw, row_date=info["date"],
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
    out_path = os.path.join(cfg["data"]["output_dir"], "predictions_ridge.xlsx")
    write_predictions(test_info, all_predictions, price_columns, out_path)

    print("\n" + "=" * 60)
    print("  Done (Ridge predictions).")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate predictions using QORC + Ridge model"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()
    main(args)
