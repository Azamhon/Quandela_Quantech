"""
Ensemble Quantum Optical Reservoir Computing (QORC) using MerLin.

Architecture per reservoir:
    input_dim → [Linear projection] → n_modes → [Sigmoid] → [0,1]
              → Phase encoding into photonic circuit (U | PS | U)
              → Fock-state probability measurement
              → (batch, fock_output_dim)  [fixed, no training]

Ensemble: 3 reservoirs with different (n_modes, n_photons, seed) configs.
Their outputs are concatenated → rich quantum feature vector.

All circuit weights are FIXED (Haar-random unitaries, not trained).
Only the downstream classical head (hybrid_model.py) is trained.
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn

# MerLin imports — available on the training machine after `pip install merlinquantum`
try:
    import perceval as pcvl
    import merlin as ML
    MERLIN_AVAILABLE = True
except ImportError:
    MERLIN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Fock output size helper
# ─────────────────────────────────────────────────────────────

def fock_output_size(n_modes, n_photons, use_fock=True):
    """
    Number of Fock-state probabilities output by one reservoir.
    use_fock=True  → full Fock space (bunching allowed): C(n+m-1, n)
    use_fock=False → no-bunching subspace:               C(m, n)
    """
    if use_fock:
        return math.comb(n_photons + n_modes - 1, n_photons)
    else:
        return math.comb(n_modes, n_photons)


# ─────────────────────────────────────────────────────────────
# Single QORC reservoir layer
# ─────────────────────────────────────────────────────────────

def _build_qorc_circuit(n_modes, n_photons, seed, device, use_fock=True):
    """
    Build one fixed QORC reservoir using MerLin.

    Circuit: U_random | PhaseShifters(px1..pxM) | U_random
    Input state: n_photons evenly spread across modes.

    Returns:
        layer:       ML.QuantumLayer (eval mode, no grad)
        output_dim:  int — number of Fock probabilities
    """
    assert MERLIN_AVAILABLE, (
        "MerLin is not installed. Run: pip install merlinquantum"
    )

    np.random.seed(seed)

    # ── Haar-random interferometer (same matrix used twice = sandwich) ──
    U_mat  = pcvl.Matrix.random_unitary(n_modes)
    U1     = pcvl.Unitary(U_mat)
    U2     = pcvl.Unitary(U_mat.copy())

    # ── Phase-shifter column for data encoding ──
    c_var = pcvl.Circuit(n_modes)
    for i in range(n_modes):
        c_var.add(i, pcvl.PS(pcvl.P(f"px{i + 1}")))

    # ── Full circuit: U1 → PS → U2 ──
    circuit = U1 // c_var // U2

    # ── Evenly spaced photon input state ──
    input_state = [0] * n_modes
    if n_photons == 1:
        input_state[0] = 1
    else:
        step = (n_modes - 1) / (n_photons - 1)
        for k in range(n_photons):
            input_state[int(round(k * step))] = 1

    # ── Measurement strategy ──
    if use_fock:
        strategy = ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK
        )
    else:
        strategy = ML.MeasurementStrategy.probs()   # UNBUNCHED by default

    # ── QuantumLayer ──
    layer = ML.QuantumLayer(
        input_size=n_modes,
        circuit=circuit,
        trainable_parameters=[],        # fixed reservoir — nothing is trained
        input_parameters=["px"],        # "px" prefix links to px1..pxN
        input_state=input_state,
        measurement_strategy=strategy,
        device=torch.device(device),
    )
    layer.eval()
    for p in layer.parameters():
        p.requires_grad_(False)

    output_dim = fock_output_size(n_modes, n_photons, use_fock)
    return layer, output_dim


# ─────────────────────────────────────────────────────────────
# Ensemble QORC module
# ─────────────────────────────────────────────────────────────

class EnsembleQORC(nn.Module):
    """
    Ensemble of QORC reservoirs.

    Each reservoir has:
      - A fixed orthogonal linear projection: input_dim → n_modes
      - A Sigmoid to map projected features to [0, 1] (phase encoding range)
      - A fixed photonic QORC circuit (MerLin QuantumLayer)

    The Fock-probability outputs of all reservoirs are concatenated.

    Args:
        input_dim: dimension of input features (latent + temporal)
        configs:   list of dicts, each with keys:
                     n_modes, n_photons, seed
        use_fock:  True = full Fock space (recommended)
        device:    "cuda" or "cpu"
    """

    def __init__(self, input_dim, configs, use_fock=True, device="cuda"):
        super().__init__()
        self.input_dim  = input_dim
        self.configs    = configs
        self.use_fock   = use_fock
        self.device_str = device

        self.projections  = nn.ModuleList()
        self.output_dims  = []

        if MERLIN_AVAILABLE:
            self.reservoirs = []          # plain list — QuantumLayer is not nn.Module
            self._build_reservoirs()
        else:
            # Fallback mode: random Gaussian features (for offline testing only)
            self.reservoirs = None
            for cfg in configs:
                proj = self._make_projection(cfg["n_modes"], seed=cfg["seed"])
                self.projections.append(proj)
                self.output_dims.append(fock_output_size(
                    cfg["n_modes"], cfg["n_photons"], use_fock
                ))
            print(
                "[WARNING] MerLin not found — EnsembleQORC running in FALLBACK mode "
                "(random Gaussian features). Install merlinquantum for real quantum features."
            )

    def _make_projection(self, n_modes, seed=0):
        """Fixed orthogonal linear projection: input_dim → n_modes.
        Uses a dedicated seed so the result is identical in train and predict."""
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed + 10000)  # offset to avoid collision with other seeds
        proj = nn.Linear(self.input_dim, n_modes, bias=False)
        nn.init.orthogonal_(proj.weight)
        proj.weight.requires_grad_(False)
        torch.random.set_rng_state(rng_state)  # restore global RNG
        return proj

    def _build_reservoirs(self):
        """Build all QORC layers (requires MerLin)."""
        for cfg in self.configs:
            proj = self._make_projection(cfg["n_modes"], seed=cfg["seed"])
            self.projections.append(proj)

            layer, out_dim = _build_qorc_circuit(
                n_modes=cfg["n_modes"],
                n_photons=cfg["n_photons"],
                seed=cfg["seed"],
                device=self.device_str,
                use_fock=self.use_fock,
            )
            self.reservoirs.append(layer)
            self.output_dims.append(out_dim)

    @property
    def total_output_dim(self):
        return sum(self.output_dims)

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim)  — latent + temporal features, any scale
        Returns:
            features: (batch, total_output_dim)  — concatenated Fock probabilities
        """
        parts = []

        for i, (proj, out_dim) in enumerate(zip(self.projections, self.output_dims)):
            # Project to n_modes dimensions, then map to [0, 2*pi] for phase encoding
            projected = torch.sigmoid(proj(x)) * (2 * math.pi)   # (batch, n_modes)

            if MERLIN_AVAILABLE and self.reservoirs is not None:
                layer = self.reservoirs[i]
                q_feats = layer(projected)         # (batch, fock_dim)
            else:
                # Fallback: random nonlinear features (sine + random weights)
                q_feats = self._fallback_features(projected, out_dim)

            parts.append(q_feats)

        return torch.cat(parts, dim=-1)           # (batch, total_output_dim)

    def _fallback_features(self, x, out_dim):
        """
        Deterministic random nonlinear features as a drop-in replacement
        when MerLin is unavailable. NOT quantum — for dev/testing only.
        """
        device = x.device
        gen = torch.Generator(device=device).manual_seed(999)
        W = torch.randn(x.shape[-1], out_dim, device=device, generator=gen)
        b = torch.rand(out_dim, device=device, generator=gen) * 2 * math.pi
        feats = torch.sin(x @ W + b)
        # Shift + normalize to look like probabilities (positive, roughly summing to 1)
        feats = feats ** 2
        feats = feats / (feats.sum(dim=-1, keepdim=True) + 1e-8)
        return feats.detach()


# ─────────────────────────────────────────────────────────────
# Feature normalizer (fit on training quantum features)
# ─────────────────────────────────────────────────────────────

class QuantumFeatureNormalizer:
    """
    StandardScaler for quantum features (fit on train only).
    Pure numpy — no sklearn dependency.
    """

    def __init__(self):
        self.mean_ = None
        self.std_  = None
        self.is_fitted = False

    def fit_transform(self, features):
        """features: np.ndarray (N, total_fock_dim)"""
        self.mean_ = features.mean(axis=0)
        self.std_  = features.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0      # avoid division by near-zero
        self.is_fitted = True
        return ((features - self.mean_) / self.std_).astype(np.float32)

    def transform(self, features):
        assert self.is_fitted
        return ((features - self.mean_) / self.std_).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Batch feature extraction helper
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_quantum_features(ensemble, x_tensor, batch_size=64):
    """
    Extract quantum features for a large dataset in mini-batches
    (avoids OOM with large Fock spaces).

    Args:
        ensemble:   EnsembleQORC (already on correct device)
        x_tensor:   torch.Tensor (N, input_dim)
        batch_size: int
    Returns:
        np.ndarray (N, total_output_dim)
    """
    ensemble.eval()
    parts = []
    for start in range(0, len(x_tensor), batch_size):
        batch = x_tensor[start : start + batch_size]
        feats = ensemble(batch)
        parts.append(feats.cpu().numpy())
    return np.concatenate(parts, axis=0)


# ─────────────────────────────────────────────────────────────
# Default ensemble configs (from config.yaml)
# ─────────────────────────────────────────────────────────────

DEFAULT_ENSEMBLE_CONFIGS = [
    {"n_modes": 12, "n_photons": 3, "seed": 42},  # C(14,3) = 364 features
    {"n_modes": 10, "n_photons": 4, "seed": 43},  # C(13,4) = 715 features
    {"n_modes": 16, "n_photons": 2, "seed": 44},  # C(17,2) = 136 features
]
# Total quantum features: 1215


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 60)
    print("SMOKE TEST: Ensemble QORC")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"MerLin available: {MERLIN_AVAILABLE}")

    INPUT_DIM = 120   # latent(20) * window(5) + delta(20)
    BATCH     = 8

    # 1. Output size calculation
    print("\n[1] Fock output sizes per reservoir:")
    total = 0
    for cfg in DEFAULT_ENSEMBLE_CONFIGS:
        sz = fock_output_size(cfg["n_modes"], cfg["n_photons"], use_fock=True)
        print(f"  modes={cfg['n_modes']}, photons={cfg['n_photons']} "
              f"→ C({cfg['n_photons']+cfg['n_modes']-1},{cfg['n_photons']}) = {sz}")
        total += sz
    print(f"  Total quantum features: {total}")
    assert total == 1215, f"Expected 1215, got {total}"
    print("  PASSED")

    # 2. Build ensemble (will use fallback if MerLin not installed)
    print("\n[2] Building EnsembleQORC...")
    ensemble = EnsembleQORC(
        input_dim=INPUT_DIM,
        configs=DEFAULT_ENSEMBLE_CONFIGS,
        use_fock=True,
        device=str(device),
    ).to(device)
    print(f"  total_output_dim = {ensemble.total_output_dim}")
    assert ensemble.total_output_dim == 1215

    # 3. Forward pass
    print("\n[3] Forward pass check...")
    x_dummy = torch.randn(BATCH, INPUT_DIM).to(device)
    feats = ensemble(x_dummy)
    print(f"  Input:  {x_dummy.shape}")
    print(f"  Output: {feats.shape}")
    assert feats.shape == (BATCH, 1215), f"Shape mismatch: {feats.shape}"
    assert not torch.isnan(feats).any(), "NaN in quantum features!"
    assert not torch.isinf(feats).any(), "Inf in quantum features!"
    print("  PASSED")

    # 4. Quantum feature normalizer
    print("\n[4] QuantumFeatureNormalizer check...")
    fake_features = np.random.rand(100, 1215).astype(np.float32)
    norm = QuantumFeatureNormalizer()
    normed = norm.fit_transform(fake_features)
    assert normed.shape == (100, 1215)
    assert abs(normed.mean()) < 0.1, "Mean not near zero after normalisation"
    print("  PASSED")

    # 5. extract_quantum_features helper
    print("\n[5] Batch feature extraction...")
    x_all = torch.randn(50, INPUT_DIM).to(device)
    all_feats = extract_quantum_features(ensemble, x_all, batch_size=16)
    assert all_feats.shape == (50, 1215)
    print(f"  Shape: {all_feats.shape}  PASSED")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    if not MERLIN_AVAILABLE:
        print("(ran in FALLBACK mode — re-run after `pip install merlinquantum`)")
    print("=" * 60)
