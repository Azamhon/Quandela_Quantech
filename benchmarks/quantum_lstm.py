"""
Quantum LSTM baseline using simulated variational quantum circuits.

Architecture:
    Classical LSTM gates (forget, input, cell, output) are augmented with
    Variational Quantum Circuits (VQCs).  Each VQC uses:
        1. Angle encoding  — RY(input_angle) on each qubit
        2. Variational layer — trainable RY(θ) per qubit
        3. Entangling layer — nearest-neighbour CNOT cascade
        4. Measurement — Pauli-Z expectation per qubit  ∈ [-1, 1]

    The VQC outputs are linearly projected to the LSTM hidden dimension
    before being passed through the gate activations (σ / tanh).

    Reference: Chen et al., "Quantum Long Short-Term Memory" (2020).

    Simulation note:
    ─────────────────
    The quantum circuit is simulated classically using a statevector
    approach in PyTorch (fully differentiable).  For n_qubits = 4 the
    state space has 2⁴ = 16 amplitudes — perfectly tractable on CPU.
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────
# Statevector Quantum Circuit Simulator
# ─────────────────────────────────────────────────────────────

class VariationalQuantumCircuit(nn.Module):
    """
    Differentiable VQC simulator (real-valued statevector).

    For each layer the circuit applies:
        RY(data)  →  RY(θ_trainable)  →  CNOT cascade

    All index tensors are pre-computed and registered as buffers.
    """

    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2 ** n_qubits

        # Trainable variational angles (one per qubit per layer)
        self.thetas = nn.ParameterList([
            nn.Parameter(torch.randn(n_qubits) * 0.3)
            for _ in range(n_layers)
        ])

        # ── Pre-compute index tensors ────────────────────────
        n, dim = n_qubits, self.dim

        # RY gate helper: for qubit q, pair_perm maps each basis state
        # to its partner with qubit q flipped; mask_0 selects states
        # where qubit q is |0⟩.
        for q in range(n):
            pair_perm = []
            mask_0_list = []
            for i in range(dim):
                bits = [(i >> (n - 1 - b)) & 1 for b in range(n)]
                mask_0_list.append(bits[q] == 0)
                bits[q] = 1 - bits[q]
                j = sum(b << (n - 1 - b_idx) for b_idx, b in enumerate(bits))
                pair_perm.append(j)
            self.register_buffer(
                f"pair_{q}", torch.tensor(pair_perm, dtype=torch.long)
            )
            self.register_buffer(
                f"m0_{q}", torch.tensor(mask_0_list, dtype=torch.bool)
            )

        # CNOT permutation (control=q, target=q+1)
        for q in range(n - 1):
            perm = []
            for i in range(dim):
                bits = [(i >> (n - 1 - b)) & 1 for b in range(n)]
                if bits[q] == 1:
                    bits[q + 1] = 1 - bits[q + 1]
                j = sum(b << (n - 1 - b_idx) for b_idx, b in enumerate(bits))
                perm.append(j)
            self.register_buffer(
                f"cnot_{q}", torch.tensor(perm, dtype=torch.long)
            )

        # Z-measurement signs: +1 for |0⟩, -1 for |1⟩
        for q in range(n):
            signs = torch.tensor(
                [1 - 2 * ((i >> (n - 1 - q)) & 1) for i in range(dim)],
                dtype=torch.float32,
            )
            self.register_buffer(f"zsign_{q}", signs)

    # ── Gate primitives ──────────────────────────────────────

    def _ry(self, state, qubit, theta):
        """
        Apply RY(theta) to `qubit` in the statevector.

        Args:
            state: (batch, dim) real amplitudes
            qubit: int
            theta: (batch,) rotation angles
        Returns:
            (batch, dim) new state
        """
        c = torch.cos(theta / 2).unsqueeze(1)        # (B, 1)
        s = torch.sin(theta / 2).unsqueeze(1)

        pair = getattr(self, f"pair_{qubit}")          # (dim,)
        m0   = getattr(self, f"m0_{qubit}")            # (dim,)

        paired = state[:, pair]                        # qubit-flipped partner

        # RY rotation on the 2-dim subspace:
        #   new|0⟩ =  cos(θ/2)·|0⟩ − sin(θ/2)·|1⟩
        #   new|1⟩ =  sin(θ/2)·|0⟩ + cos(θ/2)·|1⟩
        new_0 =  c * state - s * paired   # entries where qubit = 0
        new_1 =  s * paired + c * state   # entries where qubit = 1

        # m0 selects positions where qubit is 0
        mask = m0.unsqueeze(0)             # (1, dim) broadcast over batch
        return torch.where(mask, new_0, new_1)

    def _cnot(self, state, idx):
        """Apply pre-computed CNOT[idx, idx+1] permutation."""
        perm = getattr(self, f"cnot_{idx}")
        return state[:, perm]

    def _measure_z(self, state):
        """
        Pauli-Z expectations for all qubits.
        Returns: (batch, n_qubits)  ∈ [-1, +1]
        """
        probs = state ** 2
        exps = []
        for q in range(self.n_qubits):
            signs = getattr(self, f"zsign_{q}")
            exps.append((probs * signs.unsqueeze(0)).sum(dim=1))
        return torch.stack(exps, dim=1)

    # ── Forward ──────────────────────────────────────────────

    def forward(self, input_angles):
        """
        Args:
            input_angles: (batch, n_qubits) — data encoding angles
        Returns:
            (batch, n_qubits) — Z expectation values
        """
        B = input_angles.shape[0]
        device = input_angles.device

        # |00…0⟩
        state = torch.zeros(B, self.dim, device=device)
        state[:, 0] = 1.0

        for l in range(self.n_layers):
            # 1. Data encoding RY
            for q in range(self.n_qubits):
                state = self._ry(state, q, input_angles[:, q])
            # 2. Variational RY
            for q in range(self.n_qubits):
                theta_q = self.thetas[l][q].expand(B)
                state = self._ry(state, q, theta_q)
            # 3. Entangling CNOTs
            for q in range(self.n_qubits - 1):
                state = self._cnot(state, q)

        return self._measure_z(state)


# ─────────────────────────────────────────────────────────────
# Quantum LSTM Cell
# ─────────────────────────────────────────────────────────────

class QuantumLSTMCell(nn.Module):
    """
    LSTM cell whose gates are computed via VQCs instead of
    plain linear transforms.

    Gate flow (for each of forget / input / cell / output):
        [h_{t-1}, x_t] → Linear → n_qubits angles → VQC → n_qubits ⟨Z⟩
                        → Linear → hidden_dim → activation
    """

    def __init__(self, input_dim, hidden_dim, n_qubits=4, n_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        combined = input_dim + hidden_dim

        # Pre-VQC projections (one per gate)
        self.proj_f = nn.Linear(combined, n_qubits)
        self.proj_i = nn.Linear(combined, n_qubits)
        self.proj_c = nn.Linear(combined, n_qubits)
        self.proj_o = nn.Linear(combined, n_qubits)

        # VQCs (one per gate)
        self.vqc_f = VariationalQuantumCircuit(n_qubits, n_layers)
        self.vqc_i = VariationalQuantumCircuit(n_qubits, n_layers)
        self.vqc_c = VariationalQuantumCircuit(n_qubits, n_layers)
        self.vqc_o = VariationalQuantumCircuit(n_qubits, n_layers)

        # Post-VQC projections
        self.post_f = nn.Linear(n_qubits, hidden_dim)
        self.post_i = nn.Linear(n_qubits, hidden_dim)
        self.post_c = nn.Linear(n_qubits, hidden_dim)
        self.post_o = nn.Linear(n_qubits, hidden_dim)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)

        f_t = torch.sigmoid(self.post_f(self.vqc_f(self.proj_f(combined))))
        i_t = torch.sigmoid(self.post_i(self.vqc_i(self.proj_i(combined))))
        c_tilde = torch.tanh(self.post_c(self.vqc_c(self.proj_c(combined))))
        o_t = torch.sigmoid(self.post_o(self.vqc_o(self.proj_o(combined))))

        c_new = f_t * c_prev + i_t * c_tilde
        h_new = o_t * torch.tanh(c_new)
        return h_new, c_new


# ─────────────────────────────────────────────────────────────
# Full Quantum LSTM model
# ─────────────────────────────────────────────────────────────

class QuantumLSTMModel(nn.Module):
    """
    Sequence model: QuantumLSTMCell → FC → latent prediction.

    Args:
        input_dim:  per-timestep feature dim (= latent_dim = 20)
        hidden_dim: LSTM hidden size (keep small — VQC is expensive)
        output_dim: prediction dim (= latent_dim = 20)
        n_qubits:   qubits per VQC
        n_layers:   VQC depth
    """

    def __init__(self, input_dim=20, hidden_dim=32, output_dim=20,
                 n_qubits=4, n_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = QuantumLSTMCell(input_dim, hidden_dim, n_qubits, n_layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """x: (batch, seq_len, input_dim)"""
        B, T, _ = x.shape
        device = x.device
        h = torch.zeros(B, self.hidden_dim, device=device)
        c = torch.zeros(B, self.hidden_dim, device=device)
        for t in range(T):
            h, c = self.cell(x[:, t, :], h, c)
        return self.fc(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# Training routine
# ─────────────────────────────────────────────────────────────

def train_quantum_lstm(
    X_train_seq, y_train, X_val_seq, y_val, *,
    hidden_dim=32, n_qubits=4, n_layers=2,
    lr=5e-4, epochs=150, batch_size=32, patience=25,
):
    """
    Train a Quantum LSTM and return val predictions + metadata.

    Note: training is slower than classical LSTM due to VQC simulation.
    We use fewer epochs and a smaller hidden dim to compensate.
    """
    device = torch.device("cpu")
    input_dim = X_train_seq.shape[2]
    output_dim = y_train.shape[1]

    model = QuantumLSTMModel(
        input_dim, hidden_dim, output_dim, n_qubits, n_layers
    ).to(device)

    print(f"    QLSTM parameters : {model.count_parameters():,}")
    print(f"    QLSTM qubits     : {n_qubits}  layers: {n_layers}")
    print(f"    State-vector dim : {2**n_qubits}")

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
        optimizer, patience=15, factor=0.5
    )

    best_val = float("inf")
    best_state = None
    no_improve = 0
    final_epoch = 0

    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

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
        if epoch % 25 == 0:
            print(f"    QLSTM Epoch {epoch:>3d}/{epochs}  val_loss={val_loss:.6f}")

        if no_improve >= patience:
            print(f"    QLSTM early stop at epoch {epoch}")
            break

    train_time = time.perf_counter() - t0

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        predictions = model(X_vl).numpy()

    # Inference timing
    sample = X_vl[:1]
    t_inf = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            model(sample)
    inference_time = (time.perf_counter() - t_inf) / 100

    return {
        "model": model,
        "predictions": predictions,
        "val_loss": best_val,
        "train_time": train_time,
        "inference_time": inference_time,
        "n_parameters": model.count_parameters(),
        "epochs_trained": final_epoch,
    }
