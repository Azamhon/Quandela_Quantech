# Hybrid Photonic QRC for Swaption Pricing
### Q-volution Hackathon 2026 | Quandela Challenge | Team Quantech

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

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows
```

### 2. Install dependencies

```bash
pip install --upgrade pip

# PyTorch (with CUDA if available)
pip install torch torchvision

# MerLin (Quandela's photonic QML framework)
pip install merlinquantum

# Remaining dependencies
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train.py
```

Training runs 4 phases automatically:

| Phase | What happens | Typical time (RTX 4090) |
|-------|-------------|--------------------------|
| 1 | Train Sparse Denoising AE (~200 epochs) | ~2 min |
| 2 | Encode 494 timesteps to latent codes | <1 min |
| 3 | Run QORC ensemble on all windows (cached once) | ~5-15 min |
| 4 | Train ClassicalHead (~300 epochs) | ~3 min |

### 4. Generate predictions

```bash
python src/predict.py
```

Produces `outputs/predictions.xlsx` — ready for submission.

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
├── tests/                    # Unit tests
├── quantech.ipynb            # Main notebook (full walkthrough)
├── requirements.txt
└── README.md
```

---

## Configuration

All hyperparameters are in `configs/config.yaml`. Key settings:

```yaml
autoencoder:
  latent_dim: 20
  mask_ratio: 0.15
  epochs: 200

quantum_reservoir:
  ensemble:
    - {n_modes: 12, n_photons: 3, seed: 42}   # C(14,3) = 364 features
    - {n_modes: 10, n_photons: 4, seed: 43}   # C(13,4) = 715 features
    - {n_modes: 16, n_photons: 2, seed: 44}   # C(17,2) = 136 features

hybrid_model:
  window_size: 5
  hidden_dims: [256, 128]
  epochs: 300

device: "cuda"
```

---

## Key References

- Li et al. (2025) — QRC for realized volatility
- Notton et al. (2025) — Perceval Quest baselines
- Fujii & Nakajima (2017) — Foundational QRC
- [MerLin docs](https://merlinquantum.ai)
