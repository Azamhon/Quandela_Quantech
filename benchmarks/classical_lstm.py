"""
Classical LSTM baseline for swaption latent-code prediction.

Architecture:
    Input  → LSTM (2 layers, hidden_dim=64) → FC → latent prediction (20-dim)

The LSTM processes the sequence of latent codes (window_size=5 steps of
20-dim vectors) and predicts the next latent code.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    Classical LSTM for next-step latent code prediction.

    Args:
        input_dim:  feature dimension per timestep (= latent_dim = 20)
        hidden_dim: LSTM hidden state size
        num_layers: number of stacked LSTM layers
        output_dim: prediction dimension (= latent_dim = 20)
        dropout:    dropout between LSTM layers
    """

    def __init__(self, input_dim=20, hidden_dim=64, num_layers=2,
                 output_dim=20, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            pred: (batch, output_dim)
        """
        output, _ = self.lstm(x)               # (batch, seq, hidden)
        last = output[:, -1, :]                 # (batch, hidden)
        return self.fc(last)                    # (batch, output_dim)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# Training routine
# ─────────────────────────────────────────────────────────────

def train_classical_lstm(
    X_train_seq, y_train, X_val_seq, y_val, *,
    hidden_dim=64, num_layers=2, dropout=0.2,
    lr=1e-3, epochs=300, batch_size=32, patience=30,
):
    """
    Train a classical LSTM and return val predictions + metadata.

    Args:
        X_train_seq : (n_train, seq_len, input_dim) — sequence input
        y_train     : (n_train, output_dim)          — target latent codes
        X_val_seq   : (n_val, seq_len, input_dim)
        y_val       : (n_val, output_dim)

    Returns:
        dict with keys: model, predictions, val_loss, train_time,
                        inference_time, n_parameters, epochs_trained
    """
    device = torch.device("cpu")
    input_dim = X_train_seq.shape[2]
    output_dim = y_train.shape[1]

    model = LSTMModel(
        input_dim, hidden_dim, num_layers, output_dim, dropout
    ).to(device)

    X_tr = torch.tensor(X_train_seq, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_vl = torch.tensor(X_val_seq, dtype=torch.float32)
    y_vl = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5
    )

    best_val = float("inf")
    best_state = None
    no_improve = 0
    final_epoch = 0

    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_vl), y_vl).item()
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        final_epoch = epoch
        if epoch % 50 == 0:
            print(f"    LSTM Epoch {epoch:>3d}/{epochs}  val_loss={val_loss:.6f}")

        if no_improve >= patience:
            print(f"    LSTM early stop at epoch {epoch}")
            break

    train_time = time.perf_counter() - t0

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        predictions = model(X_vl).numpy()

    # Inference timing
    sample = X_vl[:1]
    t_inf = time.perf_counter()
    with torch.no_grad():
        for _ in range(200):
            model(sample)
    inference_time = (time.perf_counter() - t_inf) / 200

    return {
        "model": model,
        "predictions": predictions,
        "val_loss": best_val,
        "train_time": train_time,
        "inference_time": inference_time,
        "n_parameters": model.count_parameters(),
        "epochs_trained": final_epoch,
    }
