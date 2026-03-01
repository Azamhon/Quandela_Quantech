"""
Sparse Denoising Autoencoder for swaption surface compression.

Design choices:
- Linear encoder/decoder (better generalisation with only 494 samples)
- Denoising: random masking during training → robust to missing data
- Sparsity: L1 penalty on bottleneck → interpretable latent factors
- Each latent dim learns to represent a distinct market factor
  (level, slope, curvature, smile, etc.)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class SparseDenosingAE(nn.Module):
    """
    Sparse Denoising Autoencoder.

    Architecture:
        Encoder: input_dim → hidden[0] → hidden[1] → latent_dim  (ReLU)
        Decoder: latent_dim → hidden[1] → hidden[0] → input_dim  (ReLU + Sigmoid)

    Training signal:
        loss = MSE(reconstruction, clean_input) + sparsity_lambda * L1(latent)
    """

    def __init__(self, input_dim=224, hidden_dims=(128, 64), latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder ──────────────────────────────
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        # ELU at bottleneck: allows small negative values, prevents
        # "dying neuron" problem (ReLU → dead dims) while L1 penalty
        # still encourages sparsity
        enc_layers += [nn.Linear(prev, latent_dim), nn.ELU()]
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder ──────────────────────────────
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        """x: (batch, input_dim) → z: (batch, latent_dim)"""
        return self.encoder(x)

    def decode(self, z):
        """z: (batch, latent_dim) → x_hat: (batch, input_dim)"""
        return self.decoder(z)

    def forward(self, x, mask_ratio=0.0):
        """
        Args:
            x: (batch, input_dim)  — already normalised to [0, 1]
            mask_ratio: fraction of features to zero-out (denoising)
        Returns:
            x_hat: reconstruction, z: latent codes
        """
        if mask_ratio > 0.0 and self.training:
            mask = (torch.rand_like(x) > mask_ratio).float()
            x_masked = x * mask
        else:
            x_masked = x

        z = self.encode(x_masked)
        x_hat = self.decode(z)
        return x_hat, z

    def encode_partial(self, x):
        """
        Encode even when some features are NaN (for missing-data test rows).
        NaN positions are replaced with 0 before encoding.
        """
        x_filled = x.clone()
        x_filled[torch.isnan(x_filled)] = 0.0
        return self.encode(x_filled)


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────

class AELoss(nn.Module):
    def __init__(self, sparsity_lambda=1e-4):
        super().__init__()
        self.sparsity_lambda = sparsity_lambda
        self.mse = nn.MSELoss()

    def forward(self, x_hat, x_clean, z):
        recon_loss = self.mse(x_hat, x_clean)
        sparsity_loss = z.abs().mean()
        return recon_loss + self.sparsity_lambda * sparsity_loss, recon_loss, sparsity_loss


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class AETrainer:
    """
    Full training + validation loop for the autoencoder.

    Args:
        model:          SparseDenosingAE instance
        device:         torch.device
        mask_ratio:     denoising mask fraction during training
        sparsity_lambda: L1 sparsity coefficient
        lr:             Adam learning rate
        patience:       early-stopping patience (epochs)
    """

    def __init__(
        self,
        model,
        device,
        mask_ratio=0.15,
        sparsity_lambda=1e-4,
        lr=1e-3,
        patience=30,
    ):
        self.model = model.to(device)
        self.device = device
        self.mask_ratio = mask_ratio
        self.criterion = AELoss(sparsity_lambda).to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=15
        )
        self.patience = patience
        self.history = {"train_loss": [], "val_loss": [], "val_recon": []}

    def fit(self, prices_norm, val_split=50, batch_size=32, epochs=200, verbose=True):
        """
        Train the autoencoder.

        Args:
            prices_norm: np.ndarray (494, 224) — normalised [0,1] training data
            val_split:   number of last timesteps to hold out for validation
            batch_size:  mini-batch size
            epochs:      maximum training epochs
            verbose:     print progress
        """
        n_train = len(prices_norm) - val_split
        x_train = torch.tensor(prices_norm[:n_train], dtype=torch.float32)
        x_val   = torch.tensor(prices_norm[n_train:], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(x_train), batch_size=batch_size, shuffle=True
        )

        best_val_loss = math.inf
        best_state    = None
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            # ── Train ──
            self.model.train()
            train_total = 0.0
            for (xb,) in train_loader:
                xb = xb.to(self.device)
                self.optimizer.zero_grad()
                x_hat, z = self.model(xb, mask_ratio=self.mask_ratio)
                loss, _, _ = self.criterion(x_hat, xb, z)
                loss.backward()
                self.optimizer.step()
                train_total += loss.item() * len(xb)
            train_loss = train_total / n_train

            # ── Validate ──
            self.model.eval()
            with torch.no_grad():
                xv = x_val.to(self.device)
                x_hat_v, z_v = self.model(xv, mask_ratio=0.0)
                val_total, val_recon, _ = self.criterion(x_hat_v, xv, z_v)
                val_loss  = val_total.item()
                val_recon_v = val_recon.item()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_recon"].append(val_recon_v)

            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and epoch % 20 == 0:
                print(
                    f"  Epoch {epoch:>3d}/{epochs} | "
                    f"train={train_loss:.6f} | "
                    f"val={val_loss:.6f} | "
                    f"val_recon={val_recon_v:.6f}"
                )

            if epochs_no_improve >= self.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} (patience={self.patience})")
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        if verbose:
            print(f"  Training done. Best val_loss={best_val_loss:.6f}")

        return self.history

    @torch.no_grad()
    def encode_all(self, prices_norm):
        """Encode all samples to latent codes. Returns np.ndarray (N, latent_dim)."""
        self.model.eval()
        x = torch.tensor(prices_norm, dtype=torch.float32).to(self.device)
        z = self.model.encode(x)
        return z.cpu().numpy()

    @torch.no_grad()
    def reconstruct_all(self, prices_norm):
        """Full reconstruct. Returns np.ndarray (N, input_dim)."""
        self.model.eval()
        x = torch.tensor(prices_norm, dtype=torch.float32).to(self.device)
        x_hat, _ = self.model(x, mask_ratio=0.0)
        return x_hat.cpu().numpy()


# ─────────────────────────────────────────────
# Save / Load helpers
# ─────────────────────────────────────────────

def save_autoencoder(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  Saved AE weights → {path}")


def load_autoencoder(path, input_dim=224, hidden_dims=(128, 64), latent_dim=20, device="cpu"):
    model = SparseDenosingAE(input_dim, hidden_dims, latent_dim)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"  Loaded AE weights ← {path}")
    return model


# ─────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.preprocessing import load_train_data, SwaptionPreprocessor

    print("=" * 60)
    print("SMOKE TEST: Sparse Denoising Autoencoder")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # 1. Load + preprocess data
    print("\n[1] Loading and preprocessing data...")
    _, _, prices = load_train_data()
    preprocessor = SwaptionPreprocessor()
    prices_norm = preprocessor.fit_transform(prices)
    print(f"  Data shape: {prices_norm.shape}")

    # 2. Build model
    print("\n[2] Building autoencoder...")
    model = SparseDenosingAE(input_dim=224, hidden_dims=(128, 64), latent_dim=20)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # 3. Quick forward pass check
    print("\n[3] Forward pass check...")
    x_dummy = torch.rand(16, 224)
    x_hat, z = model(x_dummy, mask_ratio=0.15)
    print(f"  Input:         {x_dummy.shape}")
    print(f"  Latent codes:  {z.shape}  (min={z.min():.4f}, max={z.max():.4f})")
    print(f"  Reconstruction:{x_hat.shape} (min={x_hat.min():.4f}, max={x_hat.max():.4f})")
    assert x_hat.shape == x_dummy.shape, "Output shape mismatch!"
    assert z.shape == (16, 20), "Latent shape mismatch!"
    # ELU bottleneck allows small negatives in z, decoder Sigmoid still in [0,1]
    assert z.min() >= -1.0, f"Latent codes too negative: {z.min():.4f}"
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0, "Sigmoid output out of [0,1]!"
    print("  PASSED")

    # 4. Short training run (10 epochs to verify the loop works)
    print("\n[4] Short training run (10 epochs)...")
    trainer = AETrainer(
        model, device,
        mask_ratio=0.15,
        sparsity_lambda=1e-4,
        lr=1e-3,
        patience=30,
    )
    history = trainer.fit(
        prices_norm, val_split=50, batch_size=32, epochs=10, verbose=True
    )
    assert len(history["train_loss"]) == 10
    assert history["train_loss"][-1] < history["train_loss"][0], "Loss not decreasing!"
    print("  PASSED")

    # 5. Encode all samples
    print("\n[5] Encoding all samples...")
    latent_codes = trainer.encode_all(prices_norm)
    print(f"  Latent codes shape: {latent_codes.shape}")
    print(f"  Latent range:  [{latent_codes.min():.4f}, {latent_codes.max():.4f}]")
    assert latent_codes.shape == (494, 20)
    print("  PASSED")

    # 6. Full reconstruct + error
    print("\n[6] Reconstruction error (after only 10 epochs, expected high)...")
    recon = trainer.reconstruct_all(prices_norm)
    rmse = np.sqrt(np.mean((prices_norm - recon) ** 2))
    print(f"  RMSE (normalised): {rmse:.6f}")
    print("  (Train for 200 epochs to get this below 0.01)")

    # 7. encode_partial with NaN
    print("\n[7] encode_partial (missing data) check...")
    x_partial = torch.rand(4, 224)
    x_partial[0, [5, 50, 100]] = float("nan")  # simulate missing
    model.eval()
    with torch.no_grad():
        z_partial = model.encode_partial(x_partial)
    assert not torch.isnan(z_partial).any(), "NaN in latent codes!"
    print(f"  Shape: {z_partial.shape}  — no NaN PASSED")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
