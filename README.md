# Hybrid Photonic QRC — Swaption Pricing
### Q-volution Hackathon 2026 | Quandela Challenge

---

## Architecture

```
Raw Swaption Data (494 x 224)
  -> Robust Preprocessing  (winsorize, RobustScaler, MinMax)
  -> Sparse Denoising AE   (224 -> 20 latent dims)
  -> Temporal Windowing    (window = 5 steps)
  -> Ensemble QORC         (3 photonic reservoirs via MerLin)
  -> Classical MLP Head    (quantum + classical features -> latent prediction)
  -> AE Decoder            (20 -> 224 swaption prices)
```

---

## Windows / PowerShell Setup & Run Guide

All commands below are for **PowerShell** on Windows with an NVIDIA GPU.

### Step 0 — Prerequisites

Make sure you have installed:
- **Python 3.10+** from https://www.python.org/downloads/windows/
  - During install, tick **"Add Python to PATH"**
- **CUDA Toolkit 12.x** from https://developer.nvidia.com/cuda-downloads
- **Git** from https://git-scm.com/download/win

Verify in PowerShell:
```powershell
python --version       # should print Python 3.10.x or later
nvidia-smi             # should print your GPU info
git --version
```

---

### Step 1 — Copy the project to the training PC

On your Mac, zip the entire folder:
```bash
cd ~/development
zip -r hackathon.zip hackathon/
```

Copy `hackathon.zip` to the Windows PC (USB / network share / cloud) and unzip it.
In PowerShell, navigate into the folder:
```powershell
cd C:\path\to\hackathon
```

---

### Step 2 — Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If you see an execution policy error, run this first (once):
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then activate again:
```powershell
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` in your prompt.

---

### Step 3 — Install dependencies

```powershell
pip install --upgrade pip

# PyTorch with CUDA 12.x support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# MerLin (Quandela's photonic QML framework)
pip install merlinquantum

# Remaining dependencies
pip install -r requirements.txt
```

Verify GPU is visible to PyTorch:
```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Should print `True` and your GPU name (e.g. `NVIDIA GeForce RTX 4090`).

---

### Step 4 — Verify MerLin installation

```powershell
python -c "import merlin as ML; import perceval as pcvl; print('MerLin OK')"
```

---

### Step 5 — Run smoke tests (optional but recommended)

Run each module's built-in smoke test to verify everything works before
committing to a full training run:

```powershell
# Test 1: Preprocessing
python src\preprocessing.py

# Test 2: Autoencoder (short 10-epoch training run)
python src\autoencoder.py

# Test 3: Quantum reservoir (uses MerLin if installed, fallback otherwise)
python src\quantum_reservoir.py

# Test 4: Hybrid model assembly
python src\hybrid_model.py
```

Each test prints `ALL SMOKE TESTS PASSED` if everything is working.

---

### Step 6 — Train the model

Full pipeline (recommended):
```powershell
python src\train.py
```

This runs 4 phases automatically:

| Phase | What happens | Typical time (RTX 4090) |
|-------|-------------|--------------------------|
| 1 | Train Sparse Denoising AE (~200 epochs) | ~2 min |
| 2 | Encode 494 timesteps to latent codes | <1 min |
| 3 | Run QORC ensemble on all windows (cached once) | ~5-15 min |
| 4 | Train ClassicalHead (~300 epochs) | ~3 min |

Checkpoints are saved to `outputs\` after each phase.

If training is interrupted, you can skip the AE phase and resume from Phase 2:
```powershell
python src\train.py --phase hybrid
```

Monitor GPU usage in a separate PowerShell window:
```powershell
nvidia-smi -l 1
```

---

### Step 7 — Generate predictions

```powershell
python src\predict.py
```

This produces `outputs\predictions.xlsx` — ready for submission.

The file matches the required format:
- Columns: `[224 price columns | Date | Type]`
- **"Future prediction"** rows: model predicts autoregressively from last known state
- **"Missing data"** rows: denoising AE imputes the 4-6 missing values

---

### Step 8 — Tuning hyperparameters (optional)

All hyperparameters are in `configs\config.yaml`. Key ones to adjust:

```yaml
autoencoder:
  latent_dim: 20         # try 15 or 25 if reconstruction error is high
  mask_ratio: 0.15       # denoising strength (0.1 to 0.3)
  epochs: 200

quantum_reservoir:
  ensemble:
    - {n_modes: 12, n_photons: 3, seed: 42}   # C(14,3) = 364 features
    - {n_modes: 10, n_photons: 4, seed: 43}   # C(13,4) = 715 features
    - {n_modes: 16, n_photons: 2, seed: 44}   # C(17,2) = 136 features

hybrid_model:
  window_size: 5         # temporal context window (try 3 or 7)
  hidden_dims: [256, 128]
  dropout: 0.2
  epochs: 300

device: "cuda"           # change to "cpu" if no GPU
```

After changing config, re-run the full pipeline:
```powershell
python src\train.py
python src\predict.py
```

---

## Project Structure

```
hackathon/
├── src/
│   ├── preprocessing.py      # Data loading, outlier treatment, normalization
│   ├── autoencoder.py        # Sparse Denoising Autoencoder (224 -> 20 dims)
│   ├── quantum_reservoir.py  # Ensemble QORC via MerLin
│   ├── hybrid_model.py       # Full pipeline assembly + loss
│   ├── train.py              # Training loop (all 4 phases)
│   ├── predict.py            # Test predictions -> predictions.xlsx
│   └── utils.py              # Config loading, seeding, device helpers
├── configs/
│   └── config.yaml           # All hyperparameters
├── DATASETS/
│   ├── train.xlsx
│   ├── test_template.xlsx
│   └── sample_Simulated_Swaption_Price.xlsx
├── outputs/                  # Created automatically during training
│   ├── ae_weights.pt         # AE checkpoint
│   ├── head_weights.pt       # ClassicalHead checkpoint
│   ├── preprocessor.npz      # Fitted scaler params
│   ├── latent_codes.npy      # Encoded training sequence
│   ├── quantum_features.npy  # Cached QORC features
│   └── predictions.xlsx      # Final submission file
├── requirements.txt
├── PLAN.md
├── CLAUDE.md
└── README.md                 # This file
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `MerLin not found` | Run `pip install merlinquantum` inside the `.venv` |
| `CUDA out of memory` | Reduce `batch_size` in `config.yaml` |
| `Execution policy` error | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Slow quantum simulation | Reduce `n_modes` or `n_photons` in `config.yaml` |
| AE reconstruction error high | Increase `latent_dim` or `epochs` |
| `ModuleNotFoundError: src` | Always run scripts from the `hackathon\` root folder |
| `perceval` import error | `pip install perceval-quandela` (installed automatically with merlinquantum) |

---

## Key References

- **ReservoirQC.pdf** — QRC for realized volatility (Li et al., 2025)
- **PhotonicML.pdf** — Perceval Quest baselines (Notton et al., 2025)
- **OriginReservoirQ.pdf** — Foundational QRC (Fujii & Nakajima, 2017)
- MerLin docs: https://merlinquantum.ai
- MerLin reproduced papers: https://github.com/merlinquantum/reproduced_papers

---

## Related resources (from challenge brief)

- https://github.com/FyodorAmanov1/QStockPrediction
- https://github.com/merlinquantum/merlin
- https://github.com/merlinquantum/reproduced_papers
