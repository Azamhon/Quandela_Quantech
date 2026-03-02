"""
Training pipeline for QORC + Ridge Regression model.

Reuses Phases 1-3 from the existing train.py (AE training, latent extraction,
quantum feature extraction) but replaces Phase 4 with Ridge fitting.

Loads existing AE and quantum features if already computed (checks outputs/).

Usage:
    python src/train_ridge.py
    python src/train_ridge.py --config configs/config.yaml
    python src/train_ridge.py --alpha 1.0
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

# ── Project imports ───────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_config, set_seed, get_device, ensure_dir
from src.preprocessing import load_train_data, SwaptionPreprocessor
from src.autoencoder import (
    SparseDenosingAE, AETrainer, save_autoencoder, load_autoencoder,
)
from src.hybrid_model import make_windows
from src.quantum_reservoir import (
    EnsembleQORC, extract_quantum_features, QuantumFeatureNormalizer,
)
from src.ridge_model import train_ridge, search_alpha, save_ridge_model


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# Phase 1: Autoencoder (reuse or train)
# ─────────────────────────────────────────────────────────────

def ensure_autoencoder(cfg, device, prices_norm, output_dir):
    """Load existing AE or train a new one."""
    ae_cfg = cfg["autoencoder"]
    ae_path = os.path.join(output_dir, "ae_weights.pt")

    if os.path.exists(ae_path):
        print_section("Phase 1 — Loading existing AE checkpoint")
        ae_model = load_autoencoder(
            ae_path,
            input_dim=ae_cfg["input_dim"],
            hidden_dims=tuple(ae_cfg["hidden_dims"]),
            latent_dim=ae_cfg["latent_dim"],
            device=str(device),
        )
        return ae_model

    print_section("Phase 1 — Training Sparse Denoising Autoencoder")
    model = SparseDenosingAE(
        input_dim=ae_cfg["input_dim"],
        hidden_dims=tuple(ae_cfg["hidden_dims"]),
        latent_dim=ae_cfg["latent_dim"],
    )
    trainer = AETrainer(
        model, device,
        mask_ratio=ae_cfg["mask_ratio"],
        sparsity_lambda=ae_cfg["sparsity_lambda"],
        lr=ae_cfg["learning_rate"],
        patience=ae_cfg["patience"],
    )
    trainer.fit(
        prices_norm,
        val_split=ae_cfg["val_split"],
        batch_size=ae_cfg["batch_size"],
        epochs=ae_cfg["epochs"],
        verbose=True,
    )
    save_autoencoder(model, ae_path)

    recon = trainer.reconstruct_all(prices_norm)
    recon_err = math.sqrt(((recon - prices_norm) ** 2).mean())
    print(f"\n  Reconstruction RMSE (normalised): {recon_err:.6f}")
    return model


# ─────────────────────────────────────────────────────────────
# Phase 2: Latent codes (reuse or extract)
# ─────────────────────────────────────────────────────────────

def ensure_latent_codes(ae_model, prices_norm, device, output_dir):
    """Load existing latent codes or extract them."""
    latent_path = os.path.join(output_dir, "latent_codes.npy")

    if os.path.exists(latent_path):
        print_section("Phase 2 — Loading existing latent codes")
        latent_codes = np.load(latent_path)
        print(f"  Latent codes shape: {latent_codes.shape}")
        return latent_codes

    print_section("Phase 2 — Extracting Latent Codes")
    ae_model.eval()
    x = torch.tensor(prices_norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        z = ae_model.encode(x)
    latent_codes = z.cpu().numpy()
    np.save(latent_path, latent_codes)
    print(f"  Latent codes shape : {latent_codes.shape}")
    print(f"  Saved → {latent_path}")
    return latent_codes


# ─────────────────────────────────────────────────────────────
# Phase 3: Quantum features (reuse or compute)
# ─────────────────────────────────────────────────────────────

def ensure_quantum_features(cfg, device, X_context, output_dir):
    """Load existing quantum features or compute them."""
    Q_path = os.path.join(output_dir, "quantum_features.npy")

    qrc_cfg = cfg["quantum_reservoir"]
    val_split = cfg["autoencoder"]["val_split"]

    if os.path.exists(Q_path):
        print_section("Phase 3 — Loading existing quantum features")
        Q_features = np.load(Q_path)
        print(f"  Quantum features shape: {Q_features.shape}")
        return Q_features

    print_section("Phase 3 — Pre-computing Quantum Features")
    classical_dim = X_context.shape[1]
    ensemble = EnsembleQORC(
        input_dim=classical_dim,
        configs=qrc_cfg["ensemble"],
        use_fock=qrc_cfg["use_fock"],
        device=str(device),
    ).to(device)

    print(f"  Quantum output dim : {ensemble.total_output_dim}")
    print(f"  Running photonic simulation on {len(X_context)} windows...")
    t0 = time.time()

    x_tensor = torch.tensor(X_context, dtype=torch.float32).to(device)
    Q_raw = extract_quantum_features(ensemble, x_tensor, batch_size=64)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Normalise — fit only on training portion
    n_train = len(X_context) - val_split
    qf_norm = QuantumFeatureNormalizer()
    Q_train_n = qf_norm.fit_transform(Q_raw[:n_train])
    Q_val_n = qf_norm.transform(Q_raw[n_train:])
    Q_norm = np.concatenate([Q_train_n, Q_val_n], axis=0)

    np.save(Q_path, Q_norm)
    print(f"  Quantum features shape: {Q_norm.shape}")
    print(f"  Saved → {Q_path}")
    return Q_norm


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args):
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg)
    output_dir = cfg["data"]["output_dir"]
    ensure_dir(output_dir)

    print(f"\nDevice  : {device}")
    print(f"Seed    : {cfg['seed']}")

    # ── Load raw data ──────────────────────────────────────────
    print_section("Loading Data")
    _, _, prices = load_train_data(cfg["data"]["train_path"])
    print(f"  Raw prices shape: {prices.shape}")

    preprocessor = SwaptionPreprocessor(
        winsorize_limits=tuple(cfg["preprocessing"]["winsorize_limits"])
    )
    prices_norm = preprocessor.fit_transform(prices)
    print(f"  Normalised range: [{prices_norm.min():.4f}, {prices_norm.max():.4f}]")

    # Save preprocessor (if not already saved)
    prep_path = os.path.join(output_dir, "preprocessor.npz")
    if not os.path.exists(prep_path):
        np.savez(
            prep_path,
            median=preprocessor.median_,
            iqr=preprocessor.iqr_,
            min=preprocessor.min_,
            range=preprocessor.range_,
            clip_lower=preprocessor.clip_lower_,
            clip_upper=preprocessor.clip_upper_,
        )
        print(f"  Preprocessor saved → {prep_path}")

    # ── Phase 1: Autoencoder ───────────────────────────────────
    ae_model = ensure_autoencoder(cfg, device, prices_norm, output_dir)

    # ── Phase 2: Latent codes ──────────────────────────────────
    latent_codes = ensure_latent_codes(ae_model, prices_norm, device, output_dir)

    # ── Build windowed context ─────────────────────────────────
    window_size = cfg["hybrid_model"]["window_size"]
    X_context, y_latent, _ = make_windows(latent_codes, window_size=window_size)
    print(f"\n  Windows shape: {X_context.shape}")
    print(f"  Targets shape: {y_latent.shape}")

    # ── Phase 3: Quantum features ──────────────────────────────
    Q_features = ensure_quantum_features(cfg, device, X_context, output_dir)

    # ── Phase 4: Train Ridge ───────────────────────────────────
    print_section("Phase 4 — Ridge Regression on QORC Features")
    val_split = cfg["autoencoder"]["val_split"]

    if args.alpha is not None:
        # Train with specific alpha
        model, results = train_ridge(
            X_context, Q_features, y_latent, val_split,
            alpha=args.alpha, ae_model=ae_model, verbose=True,
        )
    else:
        # Search for best alpha
        model, results, all_results = search_alpha(
            X_context, Q_features, y_latent, val_split,
            alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
            ae_model=ae_model, verbose=True,
        )

    # Save
    ridge_path = os.path.join(output_dir, "ridge_model.joblib")
    save_ridge_model(model, ridge_path)

    print_section("Training Complete")
    print(f"  Ridge model saved → {ridge_path}")
    print(f"  Best alpha         : {results['alpha']}")
    print(f"  Val MSE (latent)   : {results['val_mse']:.6f}")
    print(f"  Val RMSE (latent)  : {results['val_rmse']:.6f}")
    print(f"  Val R²             : {results['r2']:.4f}")
    if results['surface_mse'] is not None:
        print(f"  Surface MSE        : {results['surface_mse']:.6f}")
        print(f"  Surface RMSE       : {results['surface_rmse']:.6f}")
    print(f"\n  Next step: python src/predict_ridge.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train QORC + Ridge Regression model"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Ridge alpha (if not set, searches [0.01, 0.1, 1.0, 10.0, 100.0])"
    )
    args = parser.parse_args()
    main(args)
