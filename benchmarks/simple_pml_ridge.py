"""
Benchmark: Simple Photonic MerLin + Ridge (no reservoir sandwich structure).

A single MerLin QuantumLayer with NO reservoir structure:
  - Just one Haar-random unitary (not sandwich) with phase shifters
  - Same photon config as the largest reservoir (12 modes, 3 photons)
  - Encode the 120-dim classical context → project to 12 modes
    → single unitary → measure Fock probs
  - Feed features into Ridge

Purpose: shows that the QORC sandwich architecture specifically matters,
not just "any quantum processing".

If MerLin not available, provides a fallback like the existing
_fallback_features in quantum_reservoir.py.
"""

import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

from sklearn.linear_model import Ridge

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import perceval as pcvl
    import merlin as ML
    MERLIN_AVAILABLE = True
except ImportError:
    MERLIN_AVAILABLE = False

from src.quantum_reservoir import fock_output_size


# ─────────────────────────────────────────────────────────────
# Simple (non-sandwich) quantum feature extractor
# ─────────────────────────────────────────────────────────────

class SimplePMLLayer(nn.Module):
    """
    Simple photonic feature extractor WITHOUT sandwich structure.

    Circuit: PhaseShifters(input) → single Haar-random Unitary → Fock measurement
    (Compare to QORC which uses: U_random → PS(input) → U_random)

    This deliberately lacks the double-unitary sandwich that makes reservoir
    computing effective, to demonstrate that architecture matters.
    """

    def __init__(self, input_dim=120, n_modes=12, n_photons=3,
                 use_fock=True, device="cpu", seed=42):
        super().__init__()
        self.input_dim = input_dim
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.use_fock = use_fock
        self.device_str = device
        self.seed = seed

        self.fock_dim = fock_output_size(n_modes, n_photons, use_fock)

        # Fixed orthogonal projection: input_dim → n_modes
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed + 10000)
        self.projection = nn.Linear(input_dim, n_modes, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.projection.weight.requires_grad_(False)
        torch.random.set_rng_state(rng_state)

        if MERLIN_AVAILABLE:
            self._build_simple_circuit()
        else:
            print("  [WARNING] MerLin not found — Simple PML running in FALLBACK mode")
            self.quantum_layer = None

    def _build_simple_circuit(self):
        """Build a simple circuit: PS(input) → U (single unitary, no sandwich)."""
        n = self.n_modes
        np.random.seed(self.seed)

        # Input phase shifters
        c_var = pcvl.Circuit(n)
        for i in range(n):
            c_var.add(i, pcvl.PS(pcvl.P(f"px{i + 1}")))

        # Single Haar-random unitary (NOT a sandwich)
        U_mat = pcvl.Matrix.random_unitary(n)
        U = pcvl.Unitary(U_mat)

        # Circuit: PS → U (no second unitary before PS)
        circuit = c_var // U

        # Input state
        input_state = [0] * n
        if self.n_photons == 1:
            input_state[0] = 1
        else:
            step = (n - 1) / (self.n_photons - 1)
            for k in range(self.n_photons):
                input_state[int(round(k * step))] = 1

        # Measurement
        if self.use_fock:
            strategy = ML.MeasurementStrategy.probs(
                computation_space=ML.ComputationSpace.FOCK
            )
        else:
            strategy = ML.MeasurementStrategy.probs()

        self.quantum_layer = ML.QuantumLayer(
            input_size=n,
            circuit=circuit,
            trainable_parameters=[],
            input_parameters=["px"],
            input_state=input_state,
            measurement_strategy=strategy,
            device=torch.device(self.device_str),
        )
        self.quantum_layer.eval()
        for p in self.quantum_layer.parameters():
            p.requires_grad_(False)

    def _fallback_features(self, x):
        """Deterministic random features fallback (not quantum)."""
        device = x.device
        torch.manual_seed(self.seed + 777)  # different seed from QORC fallback
        W = torch.randn(x.shape[-1], self.fock_dim, device=device)
        b = torch.rand(self.fock_dim, device=device) * 2 * math.pi
        # Simpler nonlinearity than QORC (no double-mixing)
        feats = torch.cos(x @ W + b)
        feats = feats ** 2
        feats = feats / (feats.sum(dim=-1, keepdim=True) + 1e-8)
        return feats.detach()

    @torch.no_grad()
    def forward(self, x):
        """
        x: (batch, input_dim)
        Returns: (batch, fock_dim)
        """
        projected = torch.sigmoid(self.projection(x))  # (batch, n_modes) in [0,1]

        if MERLIN_AVAILABLE and self.quantum_layer is not None:
            q_feats = self.quantum_layer(projected)
        else:
            q_feats = self._fallback_features(projected)

        return q_feats


# ─────────────────────────────────────────────────────────────
# Training: Simple PML features + Ridge readout
# ─────────────────────────────────────────────────────────────

def train_simple_pml_ridge(X_train, y_train, X_val, y_val,
                           n_modes=12, n_photons=3, alpha=1.0,
                           device="cpu"):
    """
    Extract features with simple (non-sandwich) photonic layer, then Ridge.

    Args:
        X_train, y_train: training data (flat context → latent targets)
        X_val, y_val: validation data
        n_modes, n_photons: circuit config
        alpha: Ridge regularisation

    Returns:
        dict with predictions, val_loss, train_time, inference_time, n_parameters
    """
    input_dim = X_train.shape[1]
    fock_dim = fock_output_size(n_modes, n_photons, use_fock=True)

    print(f"    Simple PML config: {n_modes} modes, {n_photons} photons")
    print(f"    Fock output dim: {fock_dim}")
    print(f"    MerLin available: {MERLIN_AVAILABLE}")

    # Build simple layer
    layer = SimplePMLLayer(
        input_dim=input_dim,
        n_modes=n_modes,
        n_photons=n_photons,
        use_fock=True,
        device=device,
    ).to(device)

    # Extract features
    t0 = time.perf_counter()

    X_tr_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_vl_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    # Batch extraction
    def extract_batch(layer, x_tensor, batch_size=64):
        parts = []
        for start in range(0, len(x_tensor), batch_size):
            batch = x_tensor[start:start + batch_size]
            feats = layer(batch)
            parts.append(feats.cpu().numpy())
        return np.concatenate(parts, axis=0)

    Q_train = extract_batch(layer, X_tr_t)
    Q_val = extract_batch(layer, X_vl_t)

    # Normalise quantum features
    q_mean = Q_train.mean(axis=0)
    q_std = Q_train.std(axis=0)
    q_std[q_std < 1e-8] = 1.0
    Q_train_n = (Q_train - q_mean) / q_std
    Q_val_n = (Q_val - q_mean) / q_std

    # Concatenate with classical context
    X_tr_full = np.hstack([Q_train_n, X_train])
    X_vl_full = np.hstack([Q_val_n, X_val])

    # Fit Ridge
    model = Ridge(alpha=alpha)
    model.fit(X_tr_full, y_train)
    train_time = time.perf_counter() - t0

    # Predictions
    predictions = model.predict(X_vl_full)
    val_loss = float(np.mean((predictions - y_val) ** 2))

    # Parameter count
    n_params = int(np.prod(model.coef_.shape))
    if model.intercept_ is not None:
        n_params += int(np.prod(np.array(model.intercept_).shape))

    # Inference timing
    sample_t = X_vl_t[:1]
    sample_x = X_val[:1]
    t_inf = time.perf_counter()
    for _ in range(200):
        q = layer(sample_t).cpu().numpy()
        q_n = (q - q_mean) / q_std
        x_full = np.hstack([q_n, sample_x])
        model.predict(x_full)
    inference_time = (time.perf_counter() - t_inf) / 200

    print(f"    Val MSE: {val_loss:.6f}")
    print(f"    Train time: {train_time:.2f}s")
    print(f"    Params: {n_params:,}")

    return {
        "model": model,
        "layer": layer,
        "q_mean": q_mean,
        "q_std": q_std,
        "predictions": predictions,
        "val_loss": val_loss,
        "train_time": train_time,
        "inference_time": inference_time,
        "n_parameters": n_params,
    }


# ─────────────────────────────────────────────────────────────
# Benchmark runner interface
# ─────────────────────────────────────────────────────────────

def run(latent_codes, prices_norm, ae_model, preprocessor, cfg, device):
    """
    Run Simple PML + Ridge benchmark.

    Returns dict matching the benchmark format.
    """
    from src.hybrid_model import make_windows

    window_size = cfg["hybrid_model"]["window_size"]
    val_split = cfg["autoencoder"]["val_split"]

    X_all, y_all, _ = make_windows(latent_codes, window_size=window_size)
    n_train = len(X_all) - val_split

    X_train = X_all[:n_train]
    X_val = X_all[n_train:]
    y_train = y_all[:n_train]
    y_val = y_all[n_train:]

    return train_simple_pml_ridge(
        X_train, y_train, X_val, y_val,
        n_modes=12, n_photons=3, alpha=1.0,
        device=str(device) if isinstance(device, torch.device) else device,
    )
