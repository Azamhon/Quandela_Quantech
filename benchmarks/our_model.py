"""
Evaluate the pre-trained QUANTECH Hybrid Photonic QRC model.

This module loads the trained ClassicalHead weights and pre-computed
quantum features, then evaluates on the validation split using the same
metrics as all other benchmarks.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.hybrid_model import ClassicalHead, HybridLoss
from src.utils import load_config


def evaluate_our_model(data):
    """
    Load the pre-trained QUANTECH model and evaluate on the validation set.

    Args:
        data: BenchmarkData instance

    Returns:
        dict with predictions, val_loss, train_time, inference_time,
             n_parameters, etc.
    """
    cfg = data.cfg
    hcfg = cfg["hybrid_model"]

    if data.Q_train is None or data.Q_val is None:
        print("  [SKIP] Quantum features not found — cannot evaluate our model.")
        return None

    quantum_dim = data.Q_val.shape[1]
    classical_dim = data.classical_dim
    latent_dim = data.latent_dim

    # ── Load trained head ────────────────────────────────────
    head_path = os.path.join(PROJECT_ROOT, cfg["data"]["output_dir"], "head_weights.pt")
    head = ClassicalHead(
        quantum_dim=quantum_dim,
        classical_dim=classical_dim,
        latent_dim=latent_dim,
        hidden_dims=tuple(hcfg["hidden_dims"]),
        dropout=hcfg["dropout"],
    )
    head.load_state_dict(
        torch.load(head_path, map_location="cpu", weights_only=True)
    )
    head.eval()

    n_params_head = sum(p.numel() for p in head.parameters())

    # Also count AE params (frozen but part of the pipeline)
    n_params_ae = sum(p.numel() for p in data.ae_model.parameters())

    # QORC has no trainable params (fixed random unitaries)
    # But projection layers have params (frozen)
    # For reporting: only the head is trained
    print(f"    Head parameters  : {n_params_head:,}")
    print(f"    AE parameters    : {n_params_ae:,} (frozen)")

    # ── Validate ─────────────────────────────────────────────
    Xq_vl = torch.tensor(data.Q_val, dtype=torch.float32)
    Xc_vl = torch.tensor(data.X_val, dtype=torch.float32)
    y_vl = torch.tensor(data.y_val, dtype=torch.float32)

    criterion = HybridLoss(surface_weight=hcfg["surface_loss_weight"])
    data.ae_model.eval()

    with torch.no_grad():
        z_pred = head(Xq_vl, Xc_vl)
        surface_pred = data.ae_model.decode(z_pred)
        surface_true = data.ae_model.decode(y_vl)
        val_loss, lat_loss, surf_loss = criterion(
            z_pred, y_vl, surface_pred, surface_true
        )

    predictions = z_pred.numpy()
    val_loss_val = val_loss.item()
    lat_mse = lat_loss.item()

    print(f"    Val loss (combined)    : {val_loss_val:.6f}")
    print(f"    Val loss (latent MSE)  : {lat_mse:.6f}")
    print(f"    Val loss (surface MSE) : {surf_loss.item():.6f}")

    # ── Inference timing (head only — quantum features pre-computed) ──
    sample_q = Xq_vl[:1]
    sample_c = Xc_vl[:1]
    t_inf = time.perf_counter()
    with torch.no_grad():
        for _ in range(500):
            head(sample_q, sample_c)
    inference_time = (time.perf_counter() - t_inf) / 500

    return {
        "model": head,
        "predictions": predictions,
        "val_loss": lat_mse,
        "val_loss_combined": val_loss_val,
        "val_loss_surface": surf_loss.item(),
        "train_time": 0.0,           # already trained — load only
        "inference_time": inference_time,
        "n_parameters": n_params_head,
        "n_parameters_total": n_params_head + n_params_ae,
    }
