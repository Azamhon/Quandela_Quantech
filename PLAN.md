# Implementation Plan: Hybrid Photonic QRC for Swaption Pricing

## Context
Q-volution Hackathon 2026 (Quandela): QML model using MerLin to predict swaption prices.
494 observations × 224 tenor/maturity columns. Test: 8 rows (future prediction + missing data).
Judging: 35% accuracy, 35% creativity, 30% presentation. RTX 4090 + 64GB RAM. 1-2 days.

## Architecture
```
Raw Data → Robust Preprocessing → Sparse Denoising AE → Latent codes (~20 dims)
→ Temporal Windowing → Ensemble QORC (MerLin) → Quantum Features
→ Classical MLP Head → Predicted latent codes → AE Decoder → 224 prices
```

---

## STEP 1: Environment Setup

### 1.1 Install dependencies
```bash
pip install merlinquantum torch pandas openpyxl scikit-learn pyyaml matplotlib
```

### 1.2 Clone reference repos
```bash
git clone https://github.com/merlinquantum/merlin.git /tmp/merlin
git clone https://github.com/merlinquantum/reproduced_papers.git /tmp/reproduced_papers
```

### 1.3 Create project structure
```bash
mkdir -p src configs notebooks outputs
```

### 1.4 Verify MerLin works
- Create minimal QuantumLayer, run forward pass, check output shape
- Reference: `/tmp/reproduced_papers/papers/QORC/lib/lib_qorc_encoding_and_linear_training.py`

---

## STEP 2: Data Loading & Exploration

### 2.1 Load datasets
- Read train.xlsx → extract Date column + 224 price columns
- Parse column headers `"Tenor : X; Maturity : Y"` → build tenor/maturity grid
- Read test_template.xlsx → identify "Future prediction" vs "Missing data" rows
- Read sample output → understand exact output format

### 2.2 Quick EDA
- Value ranges, distribution shapes per column
- Tenor × maturity grid dimensions (e.g., 14 tenors × 16 maturities = 224?)
- Temporal patterns: plot a few tenor/maturity combos over time

### 2.3 Understand output format
- Match sample_Simulated_Swaption_Price.xlsx structure exactly
- Note: output must include Date, Type, and all 224 price columns

---

## STEP 3: Preprocessing Pipeline (`src/preprocessing.py`)

### 3.1 Outlier detection & treatment
- Per-column IQR-based detection (1.5× IQR threshold)
- Winsorize to 1st/99th percentile (clip extremes, preserve time order)
- Log which timestamps/columns had outliers

### 3.2 Normalization
- **RobustScaler** (median/IQR) per column → resistant to remaining outliers
- **MinMaxScaler** to [0, 1] → bounded range for AE

### 3.3 Store fitted scalers
- Save all scaler parameters (median, IQR, min, max) for:
  - Test-time transform
  - Inverse transform (recover actual prices from predictions)

### 3.4 Verify invertibility
- `inverse_transform(transform(X))` ≈ X within 1e-6 tolerance

---

## STEP 4: Sparse Denoising Autoencoder (`src/autoencoder.py`)

### 4.1 Architecture design
```python
# Encoder: 224 → 128 → 64 → 20  (ReLU activations)
# Decoder: 20 → 64 → 128 → 224  (ReLU + Sigmoid final)
# latent_dim = 20 (tune: try 15, 20, 25)
```

### 4.2 Denoising mechanism
- During training: randomly mask 15% of input values (set to 0)
- Model learns to reconstruct full surface from partial input
- This directly prepares for the "Missing data" test task

### 4.3 Sparsity constraint
- L1 penalty on bottleneck activations: `lambda * ||z||_1`
- lambda = 1e-4 (tune if needed)
- Forces interpretable latent dimensions (each = a market factor)

### 4.4 Training
- **Loss:** `MSE_reconstruction + lambda * L1_sparsity(z)`
- **Optimizer:** Adam, lr=1e-3
- **Scheduler:** ReduceLROnPlateau (patience=15, factor=0.5)
- **Epochs:** ~200 (early stopping patience=30)
- **Validation:** Hold out last 50 timesteps (444 train / 50 val)

### 4.5 Quality checks
- Reconstruction error < 5% on validation
- Plot latent codes over time → should be smooth
- Check each latent dim's correlation with market factors
- PCA on latent codes: how many dims explain 95% variance?

---

## STEP 5: Quantum Reservoir Ensemble (`src/quantum_reservoir.py`)

### 5.1 Single QORC layer (MerLin pattern)
```python
import perceval as pcvl
import merlin as ML

def create_qorc_layer(n_modes, n_photons, seed, device="cuda"):
    np.random.seed(seed)
    U = pcvl.Unitary(pcvl.Matrix.random_unitary(n_modes))
    c_var = pcvl.Circuit(n_modes)
    for i in range(n_modes):
        c_var.add(i, pcvl.PS(pcvl.P(f"px{i+1}")))
    circuit = U // c_var // U.copy()

    input_state = [0] * n_modes
    step = (n_modes - 1) / (n_photons - 1) if n_photons > 1 else 0
    for k in range(n_photons):
        input_state[int(round(k * step))] = 1

    layer = ML.QuantumLayer(
        input_size=n_modes, circuit=circuit,
        trainable_parameters=[], input_parameters=["px"],
        input_state=input_state,
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK),
        device=torch.device(device))
    layer.eval()
    return layer
```

### 5.2 Input projection
- Linear projection: input_dim → n_modes (fixed, orthogonal init)
- Sigmoid activation → [0, 1] for phase encoding

### 5.3 Ensemble: 3 reservoir configs
| ID | Modes | Photons | Fock output dim | Purpose |
|----|-------|---------|-----------------|---------|
| R1 | 12 | 3 | C(14,3) = 364 | Medium resolution |
| R2 | 10 | 4 | C(13,4) = 715 | High nonlinearity |
| R3 | 16 | 2 | C(17,2) = 136 | Wide, low photon |
| **Total** | | | **1215 features** | |

### 5.4 Verification
- Each reservoir: correct output shape, probabilities sum to ~1
- Full ensemble: concatenated features, no NaN/Inf

### 5.5 Feature normalization
- StandardScaler on quantum features (fit on train only)
- Optional: variance threshold to remove near-constant features

---

## STEP 6: Hybrid Model (`src/hybrid_model.py`)

### 6.1 Temporal windowing
- Sliding windows of latent codes: window_size=5, stride=1
- Compute delta features: `Δz_t = z_t - z_{t-1}`
- Input vector = `[z_{t-4}, z_{t-3}, z_{t-2}, z_{t-1}, z_t, Δz_t]`
- Dimensions: 5 × 20 + 20 = 120

### 6.2 Classical head (MLP)
```python
class HybridHead(nn.Module):
    # Input: quantum_features (1215) + classical_features (120) = 1335
    # Layers: 1335 → 256 (GELU, Dropout 0.2) → 128 (GELU) → 20 (latent_dim)
```

### 6.3 Decoder connection
- Freeze AE decoder weights
- `prediction = decoder(head_output)` → 224-dim surface

### 6.4 Loss function
```python
L = MSE(z_pred, z_true) + alpha * MSE(surface_pred, surface_true)
# alpha = 0.1 (latent loss dominates for stability, surface loss for end accuracy)
```

---

## STEP 7: Training (`src/train.py`)

### 7.1 Data preparation
- Encode all 494 timesteps → latent codes (frozen encoder)
- Create sliding windows → ~489 training samples
- Split: first ~400 train, last ~89 validation
- DataLoader with batch_size=32

### 7.2 Training loop
- **Optimizer:** Adam, lr=1e-3
- **Scheduler:** ReduceLROnPlateau (patience=20, factor=0.5)
- **Gradient clipping:** max_norm=1.0
- **Epochs:** ~300 (early stopping patience=30)
- **Regularization:** Dropout 0.2, optional weight decay 1e-5

### 7.3 Evaluation metrics
- RMSE, MAE, MAPE on validation set (full 224-dim prices)
- Per-tenor and per-maturity error breakdown
- Plot predicted vs actual surfaces

---

## STEP 8: Test Predictions (`src/predict.py`)

### 8.1 "Future prediction" rows
- Start from last 5 training latent codes
- Predict next latent code → decode → full surface
- Multi-step ahead: autoregressive (use predictions as input)
- Inverse-transform to recover actual prices

### 8.2 "Missing data" rows
- Approach A: Pass partial data through denoising AE (trained with masking)
- Approach B: Encode available columns, use QRC + head to predict full surface
- Use whichever gives better reconstruction

### 8.3 Format output
- Match test_template.xlsx structure exactly (Date, Type, 224 price columns)
- Save as .xlsx

---

## STEP 9: Classical Baseline (for comparison)

### 9.1 Build classical-only model
- Same pipeline but skip quantum reservoir
- AE latent codes → MLP head → predict
- Quantify improvement from quantum features

### 9.2 Comparison table
- RMSE, MAE, MAPE: Classical vs Hybrid QRC
- Include in presentation

---

## STEP 10: Presentation & Documentation

### 10.1 Results notebook
- Architecture diagram
- Training curves
- Surface prediction heatmaps
- Quantum vs classical comparison
- Latent space visualization (what each dim represents)

### 10.2 Novelty highlights
- Ensemble QORC on autoencoder latent manifold (unique combination)
- Sparse interpretable latent space = market factor decomposition
- Denoising AE naturally handles missing data task
- Multi-config quantum reservoirs for feature diversity

### 10.3 Code quality
- Clear README, docstrings, type hints
- Reproducible: fixed seeds, config files

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| MerLin install fails | Use Perceval directly (QORC code shows exact pattern) |
| Simulation too slow | Reduce to 2 reservoirs or fewer modes/photons |
| Overfitting (494 samples) | Dropout, L1/L2 regularization, early stopping |
| AE underfits | Increase latent_dim, add hidden layers |
| Time pressure | Steps 1-5 are critical; 6-8 can be simplified (ridge regression instead of MLP) |
| Test format mismatch | Verify against sample output file before submission |
