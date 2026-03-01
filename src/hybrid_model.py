"""
Hybrid QRC + Classical model for swaption surface prediction.

Data flow:
    latent codes (z_t)  [shape: (N, latent_dim)]
         │
         ▼
    Temporal windowing
         │  window of k consecutive latent codes + first-difference
         │  → classical_input: (batch, latent_dim*(window+1))
         ▼
    EnsembleQORC           ← fixed photonic reservoirs
         │
         │  quantum_feats: (batch, 1215)
         ▼
    Concatenate [quantum_feats | classical_input]
         │
         ▼
    ClassicalHead (MLP)    ← only trained part
         │
         │  z_pred: (batch, latent_dim)
         ▼
    AE Decoder (frozen)
         │
         ▼
    surface_pred: (batch, 224)
"""

import math
import numpy as np
import torch
import torch.nn as nn

from src.quantum_reservoir import EnsembleQORC, DEFAULT_ENSEMBLE_CONFIGS


# ─────────────────────────────────────────────────────────────
# Temporal windowing
# ─────────────────────────────────────────────────────────────

def make_windows(latent_codes, window_size=5):
    """
    Create sliding-window samples from a sequence of latent codes.

    Args:
        latent_codes: np.ndarray (N, latent_dim)
        window_size:  int — number of past steps as context

    Returns:
        X: np.ndarray (N-window_size, latent_dim*(window_size+1))
               = [z_{t-k}, ..., z_{t-1}, Δz_{t-1}]   (context + delta)
        y: np.ndarray (N-window_size, latent_dim)
               = z_t  (target: next latent code)
        indices: list of int — index t for each sample (for date alignment)
    """
    N, D = latent_codes.shape
    X, y, indices = [], [], []

    for t in range(window_size, N):
        window  = latent_codes[t - window_size : t]          # (window_size, D)
        delta   = window[-1] - window[-2]                    # first difference
        flat    = window.reshape(-1)                         # (window_size*D,)
        context = np.concatenate([flat, delta])              # (window_size*D + D,)
        X.append(context)
        y.append(latent_codes[t])
        indices.append(t)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), indices


# ─────────────────────────────────────────────────────────────
# Classical head (MLP)
# ─────────────────────────────────────────────────────────────

class ClassicalHead(nn.Module):
    """
    Small MLP that maps [quantum_features | classical_context] → next latent code.

    Args:
        quantum_dim:   dimension of concatenated Fock features
        classical_dim: dimension of windowed + delta latent context
        latent_dim:    output dimension (= AE latent_dim)
        hidden_dims:   tuple of hidden layer sizes
        dropout:       dropout probability
    """

    def __init__(
        self,
        quantum_dim,
        classical_dim,
        latent_dim,
        hidden_dims=(256, 128),
        dropout=0.2,
    ):
        super().__init__()
        in_dim = quantum_dim + classical_dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers += [nn.Linear(prev, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, quantum_feats, classical_feats):
        x = torch.cat([quantum_feats, classical_feats], dim=-1)
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# Full hybrid model
# ─────────────────────────────────────────────────────────────

class HybridQRCModel(nn.Module):
    """
    Full pipeline: EnsembleQORC + ClassicalHead.
    The AE encoder/decoder are loaded externally and passed in.

    Args:
        ae_model:          trained SparseDenosingAE (encoder + decoder frozen)
        ensemble_configs:  list of dicts for EnsembleQORC
        window_size:       temporal context window
        latent_dim:        AE bottleneck size
        hidden_dims:       ClassicalHead MLP hidden sizes
        dropout:           dropout rate
        use_fock:          full Fock space (True) vs no-bunching (False)
        device:            "cuda" or "cpu"
    """

    def __init__(
        self,
        ae_model,
        ensemble_configs=None,
        window_size=5,
        latent_dim=20,
        hidden_dims=(256, 128),
        dropout=0.2,
        use_fock=True,
        device="cuda",
    ):
        super().__init__()

        if ensemble_configs is None:
            ensemble_configs = DEFAULT_ENSEMBLE_CONFIGS

        self.window_size = window_size
        self.latent_dim  = latent_dim

        # ── Frozen AE encoder + decoder ──────────────────────
        self.ae = ae_model
        for p in self.ae.parameters():
            p.requires_grad_(False)

        # ── Classical input dimension ─────────────────────────
        # window of latent codes + one delta vector
        self.classical_dim = latent_dim * (window_size + 1)

        # ── Quantum reservoir (fixed, no grad) ────────────────
        self.quantum_ensemble = EnsembleQORC(
            input_dim=self.classical_dim,
            configs=ensemble_configs,
            use_fock=use_fock,
            device=device,
        )
        quantum_dim = self.quantum_ensemble.total_output_dim

        # ── Classical head (trainable) ────────────────────────
        self.head = ClassicalHead(
            quantum_dim=quantum_dim,
            classical_dim=self.classical_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, classical_context):
        """
        Args:
            classical_context: (batch, classical_dim)
                               = [z_{t-k}..z_{t-1} | Δz_{t-1}]
        Returns:
            surface_pred: (batch, 224)   — decoded swaption surface
            z_pred:       (batch, latent_dim) — predicted latent code
        """
        # Quantum features from the photonic reservoir
        quantum_feats = self.quantum_ensemble(classical_context)

        # MLP head predicts next latent code
        z_pred = self.head(quantum_feats, classical_context)

        # Decode to full swaption surface (frozen AE decoder)
        with torch.no_grad():
            surface_pred = self.ae.decode(z_pred)

        return surface_pred, z_pred

    def trainable_parameters(self):
        """Only the ClassicalHead parameters are trained."""
        return list(self.head.parameters())


# ─────────────────────────────────────────────────────────────
# Combined loss
# ─────────────────────────────────────────────────────────────

class HybridLoss(nn.Module):
    """
    Combined latent-space + surface-space loss.

    L = MSE(z_pred, z_true) + alpha * MSE(surface_pred, surface_true)

    Latent loss dominates early (smoother gradients).
    Surface loss provides end-to-end accuracy signal.
    """

    def __init__(self, surface_weight=0.1):
        super().__init__()
        self.surface_weight = surface_weight
        self.mse = nn.MSELoss()

    def forward(self, z_pred, z_true, surface_pred, surface_true):
        latent_loss  = self.mse(z_pred, z_true)
        surface_loss = self.mse(surface_pred, surface_true)
        total = latent_loss + self.surface_weight * surface_loss
        return total, latent_loss, surface_loss


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.autoencoder import SparseDenosingAE

    print("=" * 60)
    print("SMOKE TEST: Hybrid QRC Model")
    print("=" * 60)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LATENT_DIM = 20
    WINDOW     = 5
    BATCH      = 8
    INPUT_DIM  = 224

    print(f"\nDevice: {device}")

    # 1. make_windows
    print("\n[1] make_windows...")
    fake_latent = np.random.randn(50, LATENT_DIM).astype(np.float32)
    X, y, idx = make_windows(fake_latent, window_size=WINDOW)
    expected_classical_dim = LATENT_DIM * (WINDOW + 1)
    print(f"  X shape: {X.shape}  (expected: ({50-WINDOW}, {expected_classical_dim}))")
    print(f"  y shape: {y.shape}  (expected: ({50-WINDOW}, {LATENT_DIM}))")
    assert X.shape == (45, expected_classical_dim)
    assert y.shape == (45, LATENT_DIM)
    print("  PASSED")

    # 2. ClassicalHead forward pass
    print("\n[2] ClassicalHead forward pass...")
    head = ClassicalHead(
        quantum_dim=1215,
        classical_dim=expected_classical_dim,
        latent_dim=LATENT_DIM,
    ).to(device)
    q_dummy = torch.randn(BATCH, 1215).to(device)
    c_dummy = torch.randn(BATCH, expected_classical_dim).to(device)
    z_out   = head(q_dummy, c_dummy)
    assert z_out.shape == (BATCH, LATENT_DIM)
    print(f"  Output shape: {z_out.shape}  PASSED")

    # 3. Full HybridQRCModel forward pass
    print("\n[3] Full HybridQRCModel forward pass...")
    ae = SparseDenosingAE(INPUT_DIM, (128, 64), LATENT_DIM).to(device)
    model = HybridQRCModel(
        ae_model=ae,
        window_size=WINDOW,
        latent_dim=LATENT_DIM,
        device=str(device),
    ).to(device)

    # Verify only head parameters are trainable
    trainable = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {n_trainable:,} / {n_total:,}")

    ctx = torch.randn(BATCH, expected_classical_dim).to(device)
    surface_pred, z_pred = model(ctx)
    assert surface_pred.shape == (BATCH, INPUT_DIM)
    assert z_pred.shape == (BATCH, LATENT_DIM)
    assert surface_pred.min() >= 0.0 and surface_pred.max() <= 1.0, \
        "Decoder output out of [0,1]"
    print(f"  surface_pred: {surface_pred.shape}  z_pred: {z_pred.shape}  PASSED")

    # 4. Loss function
    print("\n[4] HybridLoss check...")
    criterion = HybridLoss(surface_weight=0.1)
    z_true       = torch.randn(BATCH, LATENT_DIM).to(device)
    surface_true = torch.rand(BATCH, INPUT_DIM).to(device)
    total, l_lat, l_surf = criterion(z_pred, z_true, surface_pred, surface_true)
    print(f"  total={total.item():.6f}  latent={l_lat.item():.6f}  "
          f"surface={l_surf.item():.6f}")
    assert not torch.isnan(total), "NaN in loss!"
    print("  PASSED")

    # 5. Backward pass through head only
    print("\n[5] Backward pass (head only)...")
    opt = torch.optim.Adam(model.trainable_parameters(), lr=1e-3)
    opt.zero_grad()
    total.backward()
    opt.step()
    grad_norms = [p.grad.norm().item() for p in model.head.parameters()
                  if p.grad is not None]
    print(f"  Head grad norms (first 3): {[f'{g:.4f}' for g in grad_norms[:3]]}")
    assert len(grad_norms) > 0, "No gradients in head!"
    print("  PASSED")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
