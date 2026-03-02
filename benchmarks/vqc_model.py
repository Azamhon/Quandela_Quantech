"""
Benchmark: Parametrized Quantum Circuit (VQC) for swaption latent prediction.

A simple variational quantum circuit where circuit parameters ARE trained
via gradient descent.

Uses MerLin's ML.QuantumLayer with trainable_parameters (not empty like QORC).
Circuit: parameterized unitaries + input encoding phase shifters.
Small circuit: 6 modes, 2 photons (keep simulation fast).

This demonstrates what happens when you try to train quantum parameters
on small data — expected: worse than fixed reservoir + Ridge.

If MerLin not available, falls back to a simple parameterized PyTorch model.
"""

import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# MerLin imports
try:
    import perceval as pcvl
    import merlin as ML
    MERLIN_AVAILABLE = True
except ImportError:
    MERLIN_AVAILABLE = False

from src.quantum_reservoir import fock_output_size


# ─────────────────────────────────────────────────────────────
# VQC model with MerLin
# ─────────────────────────────────────────────────────────────

class VQCModel(nn.Module):
    """
    Variational Quantum Circuit model using MerLin.

    Architecture:
        input (120) → Linear projection → 6 modes → sigmoid → [0,1]
        → MerLin QuantumLayer (trainable unitaries + input phase shifters)
        → Fock probabilities → Linear readout → latent (20)

    The key difference from QORC: here the circuit parameters ARE trained.
    """

    def __init__(self, input_dim=120, latent_dim=20, n_modes=6, n_photons=2,
                 device="cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.device_str = device

        self.fock_dim = fock_output_size(n_modes, n_photons, use_fock=True)

        # Input projection (trainable)
        self.projection = nn.Linear(input_dim, n_modes, bias=False)
        nn.init.orthogonal_(self.projection.weight)

        if MERLIN_AVAILABLE:
            self._build_merlin_layer()
        else:
            print("  [WARNING] MerLin not found — VQC running in FALLBACK mode")
            self.quantum_layer = None
            # Fallback: trainable nonlinear feature extractor
            self.fallback_W = nn.Parameter(torch.randn(n_modes, self.fock_dim) * 0.1)
            self.fallback_b = nn.Parameter(torch.rand(self.fock_dim) * 2 * math.pi)

        # Classical readout
        self.readout = nn.Linear(self.fock_dim, latent_dim)

    def _build_merlin_layer(self):
        """Build MerLin QuantumLayer with trainable parameters."""
        n = self.n_modes

        # Build a parameterized circuit:
        # PhaseShifters(trainable) → Unitary(random, fixed) → PhaseShifters(input) → Unitary(random, fixed)
        np.random.seed(42)

        # Fixed Haar-random interferometers
        U_mat = pcvl.Matrix.random_unitary(n)
        U1 = pcvl.Unitary(U_mat)
        U2 = pcvl.Unitary(U_mat.copy())

        # Trainable phase shifters (parameters named "th1", "th2", ...)
        c_train = pcvl.Circuit(n)
        for i in range(n):
            c_train.add(i, pcvl.PS(pcvl.P(f"th{i + 1}")))

        # Input encoding phase shifters
        c_input = pcvl.Circuit(n)
        for i in range(n):
            c_input.add(i, pcvl.PS(pcvl.P(f"px{i + 1}")))

        # Full circuit: trainable PS → U1 → input PS → U2
        circuit = c_train // U1 // c_input // U2

        # Input state
        input_state = [0] * n
        if self.n_photons == 1:
            input_state[0] = 1
        else:
            step = (n - 1) / (self.n_photons - 1)
            for k in range(self.n_photons):
                input_state[int(round(k * step))] = 1

        # Measurement
        strategy = ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK
        )

        # QuantumLayer with TRAINABLE parameters
        self.quantum_layer = ML.QuantumLayer(
            input_size=n,
            circuit=circuit,
            trainable_parameters=["th"],    # "th" prefix → th1..thN are trained
            input_parameters=["px"],        # "px" prefix → input encoding
            input_state=input_state,
            measurement_strategy=strategy,
            device=torch.device(self.device_str),
        )
        # Don't freeze — these ARE trained
        for p in self.quantum_layer.parameters():
            p.requires_grad_(True)

    def forward(self, x):
        """
        x: (batch, input_dim)
        Returns: (batch, latent_dim)
        """
        projected = torch.sigmoid(self.projection(x))  # (batch, n_modes) in [0,1]

        if MERLIN_AVAILABLE and self.quantum_layer is not None:
            q_feats = self.quantum_layer(projected)  # (batch, fock_dim)
        else:
            # Fallback: trainable sine features
            raw = projected @ self.fallback_W + self.fallback_b
            q_feats = torch.sin(raw) ** 2
            q_feats = q_feats / (q_feats.sum(dim=-1, keepdim=True) + 1e-8)

        return self.readout(q_feats)  # (batch, latent_dim)


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def train_vqc(X_train, y_train, X_val, y_val,
              n_modes=6, n_photons=2, epochs=50, lr=0.01,
              batch_size=32, device="cpu"):
    """
    Train VQC model and return results dict.

    Args:
        X_train, y_train: training data (flat context → latent targets)
        X_val, y_val: validation data
        n_modes, n_photons: circuit configuration
        epochs: number of training epochs
        lr: learning rate
        batch_size: mini-batch size
        device: "cpu" or "cuda"

    Returns:
        dict with predictions, val_loss, train_time, inference_time, n_parameters
    """
    print(f"    VQC config: {n_modes} modes, {n_photons} photons")
    print(f"    Fock output dim: {fock_output_size(n_modes, n_photons, True)}")
    print(f"    MerLin available: {MERLIN_AVAILABLE}")

    input_dim = X_train.shape[1]
    latent_dim = y_train.shape[1]

    model = VQCModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_modes=n_modes,
        n_photons=n_photons,
        device=device,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable params: {n_params:,}")

    # Data loaders
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    X_vl_t = torch.tensor(X_val, dtype=torch.float32)
    y_vl_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    t0 = time.perf_counter()
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        n_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += loss.item() * len(xb)
            n_samples += len(xb)

        train_loss = train_total / n_samples

        # Validate
        model.eval()
        with torch.no_grad():
            xv = X_vl_t.to(device)
            yv = y_vl_t.to(device)
            pred_v = model(xv)
            val_loss = criterion(pred_v, yv).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3d}/{epochs} | "
                  f"train={train_loss:.6f} | val={val_loss:.6f}")

    train_time = time.perf_counter() - t0

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_vl_t.to(device)).cpu().numpy()

    # Inference timing
    sample = X_vl_t[:1].to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):  # warmup
            model(sample)
        t_inf = time.perf_counter()
        for _ in range(200):
            model(sample)
        inference_time = (time.perf_counter() - t_inf) / 200

    print(f"    Best val MSE: {best_val_loss:.6f}")
    print(f"    Train time: {train_time:.1f}s")

    return {
        "model": model,
        "predictions": predictions,
        "val_loss": best_val_loss,
        "train_time": train_time,
        "inference_time": inference_time,
        "n_parameters": n_params,
    }


# ─────────────────────────────────────────────────────────────
# Benchmark runner interface (matches existing pattern)
# ─────────────────────────────────────────────────────────────

def run(latent_codes, prices_norm, ae_model, preprocessor, cfg, device):
    """
    Run VQC benchmark.

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

    return train_vqc(
        X_train, y_train, X_val, y_val,
        n_modes=6, n_photons=2,
        epochs=50, lr=0.01,
        batch_size=32,
        device=str(device) if isinstance(device, torch.device) else device,
    )
