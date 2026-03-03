"""
Shared data loading and preparation for all benchmarks.

Loads the pre-trained autoencoder, preprocesses raw data, creates
train/val splits identical to the main training pipeline so every
benchmark sees exactly the same samples.

Optionally loads test.xlsx as held-out ground truth for final evaluation.
"""

import os
import sys
import numpy as np
import torch

# ── Project imports ──────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_train_data, SwaptionPreprocessor
from src.autoencoder import SparseDenosingAE, load_autoencoder
from src.hybrid_model import make_windows
from src.utils import load_config, set_seed


class BenchmarkData:
    """
    Central data provider for all benchmark models.

    Loads raw prices, preprocesses them, encodes to latent space via the
    pre-trained AE, creates temporal windows, and provides train/val splits
    identical to the main pipeline (last `val_split` windows for validation).

    Attributes
    ----------
    prices_norm : np.ndarray (N, 224)
        Normalised swaption prices.
    latent_codes : np.ndarray (N, 20)
        AE-encoded latent representations.
    X_train, X_val : np.ndarray
        Flattened windowed context features  (n, 120).
    y_train, y_val : np.ndarray
        Target next-step latent codes  (n, 20).
    X_train_seq, X_val_seq : np.ndarray
        Sequence-format inputs for LSTMs  (n, window=5, 20).
    Q_train, Q_val : np.ndarray or None
        Pre-computed quantum features for our model.
    ae_model : SparseDenosingAE
        Frozen autoencoder (for surface-level evaluation).
    """

    def __init__(self, config_path="configs/config.yaml"):
        self.cfg = load_config(os.path.join(PROJECT_ROOT, config_path))
        set_seed(self.cfg["seed"])
        self.device = torch.device("cpu")  # CPU for fair comparison

        self._load_data()
        self._load_ae()
        self._prepare_splits()

    # ── internal helpers ─────────────────────────────────────

    def _load_data(self):
        train_path = os.path.join(PROJECT_ROOT, self.cfg["data"]["train_path"])
        _, self.price_columns, self.prices_raw = load_train_data(train_path)

        self.preprocessor = SwaptionPreprocessor(
            winsorize_limits=tuple(self.cfg["preprocessing"]["winsorize_limits"])
        )
        self.prices_norm = self.preprocessor.fit_transform(self.prices_raw)
        print(f"  Raw data shape     : {self.prices_raw.shape}")
        print(f"  Normalised range   : [{self.prices_norm.min():.4f}, "
              f"{self.prices_norm.max():.4f}]")

    def _load_ae(self):
        ae_cfg = self.cfg["autoencoder"]
        ae_path = os.path.join(
            PROJECT_ROOT, self.cfg["data"]["output_dir"], "ae_weights.pt"
        )
        self.ae_model = load_autoencoder(
            ae_path,
            input_dim=ae_cfg["input_dim"],
            hidden_dims=tuple(ae_cfg["hidden_dims"]),
            latent_dim=ae_cfg["latent_dim"],
            device="cpu",
        )
        self.ae_model.eval()
        for p in self.ae_model.parameters():
            p.requires_grad_(False)

        # Encode all prices to latent codes
        with torch.no_grad():
            x = torch.tensor(self.prices_norm, dtype=torch.float32)
            self.latent_codes = self.ae_model.encode(x).numpy()
        print(f"  Latent codes shape : {self.latent_codes.shape}")

    def _prepare_splits(self):
        window_size = self.cfg["hybrid_model"]["window_size"]
        val_split = self.cfg["autoencoder"]["val_split"]
        latent_dim = self.cfg["autoencoder"]["latent_dim"]

        # Windowed features (flat)
        X_all, y_all, self.indices = make_windows(
            self.latent_codes, window_size=window_size
        )
        n_total = len(X_all)
        n_train = n_total - val_split

        self.X_train = X_all[:n_train]
        self.X_val = X_all[n_train:]
        self.y_train = y_all[:n_train]
        self.y_val = y_all[n_train:]

        # Sequence format for LSTMs: (batch, seq_len, features)
        self.X_train_seq = self.X_train[:, :window_size * latent_dim].reshape(
            -1, window_size, latent_dim
        )
        self.X_val_seq = self.X_val[:, :window_size * latent_dim].reshape(
            -1, window_size, latent_dim
        )

        # Pre-computed quantum features (for our model evaluation)
        qf_path = os.path.join(
            PROJECT_ROOT, self.cfg["data"]["output_dir"], "quantum_features.npy"
        )
        if os.path.exists(qf_path):
            Q_all = np.load(qf_path)
            self.Q_train = Q_all[:n_train]
            self.Q_val = Q_all[n_train:]
        else:
            self.Q_train = None
            self.Q_val = None

        self.n_train = n_train
        self.val_split = val_split
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.classical_dim = X_all.shape[1]

        print(f"  Train samples      : {self.n_train}")
        print(f"  Val   samples      : {self.val_split}")
        print(f"  Classical dim      : {self.classical_dim}")
        print(f"  Sequence shape     : ({window_size}, {latent_dim})")
        if self.Q_train is not None:
            print(f"  Quantum feat dim   : {self.Q_train.shape[1]}")

    # ── public helpers ───────────────────────────────────────

    def decode_latent(self, z):
        """Decode latent codes back to 224-dim normalised prices."""
        with torch.no_grad():
            if isinstance(z, np.ndarray):
                z = torch.tensor(z, dtype=torch.float32)
            return self.ae_model.decode(z).numpy()

    def summary(self):
        """Print data summary."""
        print("\n" + "=" * 50)
        print("  BENCHMARK DATA SUMMARY")
        print("=" * 50)
        print(f"  Dataset          : {self.prices_raw.shape[0]} timesteps × "
              f"{self.prices_raw.shape[1]} prices")
        print(f"  AE latent dim    : {self.latent_dim}")
        print(f"  Window size      : {self.window_size}")
        print(f"  Train / Val      : {self.n_train} / {self.val_split}")
        if hasattr(self, 'n_test') and self.n_test > 0:
            print(f"  Test  samples    : {self.n_test}")
        print(f"  Input dim (flat) : {self.classical_dim}")
        print(f"  Target dim       : {self.latent_dim}")
        print("=" * 50)

    # ── Test data (held-out ground truth) ────────────────────

    def load_test_data(self, test_path=None):
        """
        Load test.xlsx as held-out ground truth for final evaluation.

        The test data is NEVER used for training or fitting — only for
        comparing model predictions against actual future prices.

        Walk-forward evaluation: for each test day, the context window
        uses only data that would have been observed at that point in time
        (training data + any preceding test days).

        Sets: X_test, y_test, X_test_seq, Q_test, n_test,
              test_prices_raw, test_prices_norm, test_latent
        """
        if test_path is None:
            test_path = os.path.join(PROJECT_ROOT, "DATASETS/test.xlsx")

        if not os.path.exists(test_path):
            print(f"  [WARN] Test file not found: {test_path}")
            self.n_test = 0
            return

        print(f"\n  Loading test data: {test_path}")

        # Load raw test prices (same format as train)
        _, _, self.test_prices_raw = load_train_data(test_path)
        n_test_rows = self.test_prices_raw.shape[0]
        print(f"  Test rows          : {n_test_rows}")

        # Transform with FITTED preprocessor — NO re-fitting (no data leakage)
        self.test_prices_norm = self.preprocessor.transform(self.test_prices_raw)
        print(f"  Test norm range    : [{self.test_prices_norm.min():.4f}, "
              f"{self.test_prices_norm.max():.4f}]")

        # Encode through frozen AE (same weights used for training data)
        with torch.no_grad():
            x = torch.tensor(self.test_prices_norm, dtype=torch.float32)
            self.test_latent = self.ae_model.encode(x).numpy()
        print(f"  Test latent shape  : {self.test_latent.shape}")

        # ── Walk-forward windows ─────────────────────────────
        # Concatenate train + test latent codes.  Windows spanning
        # the boundary use the last training days as context to predict
        # the first test day, then shift forward using actual observed
        # test values (standard walk-forward — no leakage).
        all_latent = np.concatenate([self.latent_codes, self.test_latent], axis=0)

        X_all, y_all, _ = make_windows(all_latent, window_size=self.window_size)

        # The last n_test_rows windows are the test targets
        self.X_test = X_all[-n_test_rows:]
        self.y_test = y_all[-n_test_rows:]

        # Sequence format for LSTMs
        self.X_test_seq = self.X_test[
            :, :self.window_size * self.latent_dim
        ].reshape(-1, self.window_size, self.latent_dim)

        self.n_test = n_test_rows

        # ── Quantum features for test windows ────────────────
        self.Q_test = None
        if self.Q_train is not None:
            self._compute_test_quantum_features()

        print(f"  Test windows       : {self.n_test}")
        if self.Q_test is not None:
            print(f"  Test quantum feats : {self.Q_test.shape}")

    def _compute_test_quantum_features(self):
        """
        Compute quantum features for the test windows using the same
        fixed EnsembleQORC architecture.  The reservoir weights are
        random-but-deterministic (seeded), so we get identical circuits
        as during training.
        """
        from src.quantum_reservoir import (
            EnsembleQORC, extract_quantum_features, QuantumFeatureNormalizer,
        )

        configs = self.cfg.get("quantum_reservoir", {}).get("ensemble", [
            {"n_modes": 12, "n_photons": 3, "seed": 42},
            {"n_modes": 10, "n_photons": 4, "seed": 43},
            {"n_modes": 16, "n_photons": 2, "seed": 44},
        ])
        use_fock = self.cfg.get("quantum_reservoir", {}).get("use_fock", True)

        ensemble = EnsembleQORC(
            input_dim=self.classical_dim,
            configs=configs,
            use_fock=use_fock,
            device="cpu",
        )
        ensemble.eval()

        # Extract raw quantum features for test windows
        X_test_t = torch.tensor(self.X_test, dtype=torch.float32)
        Q_test_raw = extract_quantum_features(ensemble, X_test_t, batch_size=64)

        # Normalise using statistics from training quantum features
        # (loaded from quantum_features.npy — the same features used for Q_train/Q_val)
        qf_path = os.path.join(
            PROJECT_ROOT, self.cfg["data"]["output_dir"], "quantum_features.npy"
        )
        Q_all_train = np.load(qf_path)  # all training quantum features

        # Also recompute train features to get the raw (un-normalised) versions
        # for a consistent normalization.  Since the quantum reservoir is fixed,
        # we just match the normalisation already applied to Q_train/Q_val.
        # The saved quantum_features.npy may or may not be normalised, so we
        # check by comparing shapes/stats.

        # Simpler approach: compute quantum features for a few training windows
        # to recover the normalizer parameters.  Actually, since the reservoir
        # is deterministic, let's just compute features for the full train set
        # and normalize test with the same stats.
        X_train_all_t = torch.tensor(
            np.concatenate([self.X_train, self.X_val], axis=0),
            dtype=torch.float32,
        )
        Q_train_raw = extract_quantum_features(ensemble, X_train_all_t, batch_size=64)

        normalizer = QuantumFeatureNormalizer()
        Q_train_normed = normalizer.fit_transform(Q_train_raw)
        Q_test_normed = normalizer.transform(Q_test_raw)

        self.Q_test = Q_test_normed
