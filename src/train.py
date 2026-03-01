"""
Training pipeline for the Hybrid Photonic QRC model.

Run order:
    Phase 1 — Train the Sparse Denoising Autoencoder
    Phase 2 — Extract latent codes with frozen AE encoder
    Phase 3 — Pre-compute quantum features with frozen QORC ensemble
    Phase 4 — Train the ClassicalHead on (quantum + classical) → latent

Usage:
    python src/train.py                        # use configs/config.yaml
    python src/train.py --config configs/config.yaml
    python src/train.py --phase ae             # only Phase 1
    python src/train.py --phase hybrid         # only Phases 2-4 (AE must exist)
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── project imports ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils          import load_config, set_seed, get_device, ensure_dir
from src.preprocessing  import load_train_data, SwaptionPreprocessor
from src.autoencoder    import (SparseDenosingAE, AETrainer,
                                save_autoencoder, load_autoencoder)
from src.hybrid_model   import (HybridQRCModel, make_windows, HybridLoss)
from src.quantum_reservoir import (EnsembleQORC, extract_quantum_features,
                                   QuantumFeatureNormalizer)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def rmse(pred, target):
    return math.sqrt(((pred - target) ** 2).mean())


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# Phase 1: Train Autoencoder
# ─────────────────────────────────────────────────────────────

def train_autoencoder(cfg, device, prices_norm):
    print_section("Phase 1 — Sparse Denoising Autoencoder")

    ae_cfg = cfg["autoencoder"]
    model  = SparseDenosingAE(
        input_dim   = ae_cfg["input_dim"],
        hidden_dims = tuple(ae_cfg["hidden_dims"]),
        latent_dim  = ae_cfg["latent_dim"],
    )

    trainer = AETrainer(
        model,
        device,
        mask_ratio       = ae_cfg["mask_ratio"],
        sparsity_lambda  = ae_cfg["sparsity_lambda"],
        lr               = ae_cfg["learning_rate"],
        patience         = ae_cfg["patience"],
    )

    history = trainer.fit(
        prices_norm,
        val_split  = ae_cfg["val_split"],
        batch_size = ae_cfg["batch_size"],
        epochs     = ae_cfg["epochs"],
        verbose    = True,
    )

    # Save checkpoint
    ensure_dir(cfg["data"]["output_dir"])
    ae_path = os.path.join(cfg["data"]["output_dir"], "ae_weights.pt")
    save_autoencoder(model, ae_path)

    # Report reconstruction quality
    recon     = trainer.reconstruct_all(prices_norm)
    recon_err = rmse(recon, prices_norm)
    print(f"\n  Reconstruction RMSE (normalised): {recon_err:.6f}")

    return model, trainer


# ─────────────────────────────────────────────────────────────
# Phase 2: Extract latent codes
# ─────────────────────────────────────────────────────────────

def extract_latent_codes(ae_model, prices_norm, device):
    print_section("Phase 2 — Extract Latent Codes")

    ae_model.eval()
    x = torch.tensor(prices_norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        z = ae_model.encode(x)
    latent_codes = z.cpu().numpy()

    print(f"  Latent codes shape : {latent_codes.shape}")
    print(f"  Latent range       : [{latent_codes.min():.4f}, {latent_codes.max():.4f}]")
    print(f"  Latent mean/std    : {latent_codes.mean():.4f} / {latent_codes.std():.4f}")

    return latent_codes


# ─────────────────────────────────────────────────────────────
# Phase 3: Pre-compute quantum features
# ─────────────────────────────────────────────────────────────

def precompute_quantum_features(cfg, device, X_context):
    """
    Run all latent-context windows through the frozen QORC ensemble once,
    cache the result. This avoids re-running the (slow) photonic simulation
    every epoch.

    Args:
        X_context: np.ndarray (N_windows, classical_dim)
    Returns:
        Q_features: np.ndarray (N_windows, total_fock_dim) — normalised
        qf_normalizer: QuantumFeatureNormalizer (fitted on train split)
    """
    print_section("Phase 3 — Pre-compute Quantum Features")

    qrc_cfg    = cfg["quantum_reservoir"]
    hybrid_cfg = cfg["hybrid_model"]
    val_split  = cfg["autoencoder"]["val_split"]

    classical_dim = X_context.shape[1]
    ensemble = EnsembleQORC(
        input_dim = classical_dim,
        configs   = qrc_cfg["ensemble"],
        use_fock  = qrc_cfg["use_fock"],
        device    = str(device),
    ).to(device)

    print(f"  Quantum output dim : {ensemble.total_output_dim}")
    print(f"  Running photonic simulation on {len(X_context)} windows...")
    t0 = time.time()

    x_tensor  = torch.tensor(X_context, dtype=torch.float32).to(device)
    Q_raw     = extract_quantum_features(ensemble, x_tensor, batch_size=64)

    print(f"  Done in {time.time() - t0:.1f}s")

    # Normalise — fit only on training portion
    n_train   = len(X_context) - val_split
    qf_norm   = QuantumFeatureNormalizer()
    Q_train_n = qf_norm.fit_transform(Q_raw[:n_train])
    Q_val_n   = qf_norm.transform(Q_raw[n_train:])
    Q_norm    = np.concatenate([Q_train_n, Q_val_n], axis=0)

    print(f"  Q_features shape   : {Q_norm.shape}")
    print(f"  Q_features range   : [{Q_norm.min():.4f}, {Q_norm.max():.4f}]")

    return Q_norm, qf_norm, ensemble


# ─────────────────────────────────────────────────────────────
# Phase 4: Train ClassicalHead
# ─────────────────────────────────────────────────────────────

def train_hybrid_head(cfg, device, ae_model, X_context, Q_features, y_latent):
    """
    Train the ClassicalHead on pre-computed quantum + classical features.

    The AE decoder is used in the loss (frozen) to get the surface loss term.

    Args:
        X_context  : (N, classical_dim)   — windowed latent context
        Q_features : (N, quantum_dim)     — pre-computed Fock probabilities
        y_latent   : (N, latent_dim)      — target latent codes
    """
    print_section("Phase 4 — Train Classical Head")

    hcfg       = cfg["hybrid_model"]
    ae_cfg     = cfg["autoencoder"]
    val_split  = ae_cfg["val_split"]
    n_total    = len(X_context)
    n_train    = n_total - val_split

    # ── Tensors ──
    Xc_tr = torch.tensor(X_context[:n_train],   dtype=torch.float32)
    Xq_tr = torch.tensor(Q_features[:n_train],  dtype=torch.float32)
    y_tr  = torch.tensor(y_latent[:n_train],     dtype=torch.float32)

    Xc_vl = torch.tensor(X_context[n_train:],   dtype=torch.float32)
    Xq_vl = torch.tensor(Q_features[n_train:],  dtype=torch.float32)
    y_vl  = torch.tensor(y_latent[n_train:],     dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(Xc_tr, Xq_tr, y_tr),
        batch_size=hcfg["batch_size"],
        shuffle=True,
    )

    # ── Model (head only — AE stays frozen) ──
    from src.hybrid_model import ClassicalHead
    quantum_dim   = Q_features.shape[1]
    classical_dim = X_context.shape[1]
    latent_dim    = ae_cfg["latent_dim"]

    head = ClassicalHead(
        quantum_dim   = quantum_dim,
        classical_dim = classical_dim,
        latent_dim    = latent_dim,
        hidden_dims   = tuple(hcfg["hidden_dims"]),
        dropout       = hcfg["dropout"],
    ).to(device)

    criterion = HybridLoss(surface_weight=hcfg["surface_loss_weight"])
    optimizer = torch.optim.Adam(head.parameters(), lr=hcfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )

    ae_model.eval()
    best_val  = math.inf
    best_state = None
    no_improve = 0

    for epoch in range(1, hcfg["epochs"] + 1):
        # ── Train ──
        head.train()
        train_total = 0.0
        for xc_b, xq_b, y_b in train_loader:
            xc_b, xq_b, y_b = (t.to(device) for t in (xc_b, xq_b, y_b))
            optimizer.zero_grad()

            z_pred = head(xq_b, xc_b)
            # surface_pred must be OUTSIDE no_grad so gradients flow
            # through the frozen decoder back to z_pred → head
            surface_pred = ae_model.decode(z_pred)
            with torch.no_grad():
                surface_true  = ae_model.decode(y_b)

            loss, _, _ = criterion(z_pred, y_b, surface_pred, surface_true)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), hcfg["gradient_clip"])
            optimizer.step()
            train_total += loss.item() * len(xc_b)

        train_loss = train_total / n_train

        # ── Validate ──
        head.eval()
        with torch.no_grad():
            xc_v = Xc_vl.to(device)
            xq_v = Xq_vl.to(device)
            y_v  = y_vl.to(device)
            z_pred_v      = head(xq_v, xc_v)
            surface_pred_v = ae_model.decode(z_pred_v)
            surface_true_v = ae_model.decode(y_v)
            val_loss, val_lat, val_surf = criterion(
                z_pred_v, y_v, surface_pred_v, surface_true_v
            )
            val_loss = val_loss.item()

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0:
            print(
                f"  Epoch {epoch:>3d}/{hcfg['epochs']} | "
                f"train={train_loss:.6f} | "
                f"val={val_loss:.6f} (lat={val_lat.item():.6f}, "
                f"surf={val_surf.item():.6f})"
            )

        if no_improve >= hcfg["patience"]:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state:
        head.load_state_dict(best_state)
    print(f"  Training done. Best val_loss={best_val:.6f}")

    # Save head checkpoint
    head_path = os.path.join(cfg["data"]["output_dir"], "head_weights.pt")
    torch.save(head.state_dict(), head_path)
    print(f"  Saved head → {head_path}")

    return head


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args):
    cfg    = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg)
    ensure_dir(cfg["data"]["output_dir"])

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
    print(f"  Normalised range : [{prices_norm.min():.4f}, {prices_norm.max():.4f}]")

    # Save preprocessor params for inference
    prep_path = os.path.join(cfg["data"]["output_dir"], "preprocessor.npz")
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
    ae_path = os.path.join(cfg["data"]["output_dir"], "ae_weights.pt")
    if args.phase in ("all", "ae") or not os.path.exists(ae_path):
        ae_model, _ = train_autoencoder(cfg, device, prices_norm)
    else:
        print_section("Phase 1 — Loading existing AE checkpoint")
        ae_cfg   = cfg["autoencoder"]
        ae_model = load_autoencoder(
            ae_path,
            input_dim   = ae_cfg["input_dim"],
            hidden_dims = tuple(ae_cfg["hidden_dims"]),
            latent_dim  = ae_cfg["latent_dim"],
            device      = str(device),
        )

    # ── Phase 2: Latent codes ──────────────────────────────────
    if args.phase in ("all", "ae", "hybrid"):
        latent_codes = extract_latent_codes(ae_model, prices_norm, device)
        latent_path  = os.path.join(cfg["data"]["output_dir"], "latent_codes.npy")
        np.save(latent_path, latent_codes)
        print(f"  Latent codes saved → {latent_path}")
    else:
        latent_path  = os.path.join(cfg["data"]["output_dir"], "latent_codes.npy")
        latent_codes = np.load(latent_path)
        print(f"  Loaded latent codes: {latent_codes.shape}")

    # ── Build windowed context ─────────────────────────────────
    window_size = cfg["hybrid_model"]["window_size"]
    X_context, y_latent, _ = make_windows(latent_codes, window_size=window_size)
    print(f"\n  Windows shape  : {X_context.shape}")
    print(f"  Targets shape  : {y_latent.shape}")

    # ── Phase 3: Quantum features ──────────────────────────────
    Q_path = os.path.join(cfg["data"]["output_dir"], "quantum_features.npy")
    if args.phase in ("all", "ae", "hybrid") or not os.path.exists(Q_path):
        Q_features, qf_norm, ensemble = precompute_quantum_features(
            cfg, device, X_context
        )
        np.save(Q_path, Q_features)
        print(f"  Quantum features saved → {Q_path}")
    else:
        Q_features = np.load(Q_path)
        print(f"  Loaded quantum features: {Q_features.shape}")

    # ── Phase 4: Train head ────────────────────────────────────
    head = train_hybrid_head(cfg, device, ae_model, X_context, Q_features, y_latent)

    print_section("Training Complete")
    print(f"  All outputs saved to: {cfg['data']['output_dir']}/")
    print(f"  Next step: python src/predict.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--phase", default="all",
        choices=["all", "ae", "hybrid"],
        help=(
            "all    = run all phases (default)\n"
            "ae     = train AE only\n"
            "hybrid = skip AE training, load existing checkpoint"
        ),
    )
    args = parser.parse_args()
    main(args)
