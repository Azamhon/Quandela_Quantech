# Hybrid Photonic QRC for Swaption Pricing

**Q-volution Hackathon 2026 · Quandela Challenge · Team QUANTECH**

A hybrid classical–quantum pipeline that forecasts 224-dimensional swaption price surfaces using photonic quantum reservoir computing (QRC) via Quandela's MerLin framework. The system compresses raw prices into a 20-dimensional latent space, extracts 1,215 Fock-state probability features from an ensemble of three photonic circuits, and maps them to next-day predictions with a simple Ridge readout.

---

## Architecture

```
Raw Swaption Prices (494 × 224)
  → Robust Preprocessing        Winsorize · RobustScaler · MinMax [0, 1]
  → Sparse Denoising AE         224 → 128 → 64 → 20 latent dims
  → Temporal Windowing           5-step context windows + deltas
  → Ensemble QORC               3 photonic reservoirs (MerLin)
       12 m / 3 p  → 364 Fock features
       10 m / 4 p  → 715 Fock features
       16 m / 2 p  → 136 Fock features
                       ─────────────────
                        1,215 total
  → Ridge Readout               α = 100, quantum + classical features → latent
  → AE Decoder                  20 → 64 → 128 → 224 swaption prices
```

---

## Benchmark Results

All models were trained on the same 439 windows and evaluated on both a 50-window validation split and 6 held-out future days (`test.xlsx`).

| Rank | Model | Val MSE | Test MSE | Test R² |
|------|-------|---------|----------|---------|
| 1 | **Simple PML + Ridge** | 0.0105 | **0.0085** | 0.692 |
| 2 | Ridge Regression | 0.0109 | 0.0091 | 0.673 |
| 3 | Classical LSTM | 0.0202 | 0.0096 | 0.654 |
| 4 | **★ QORC + Ridge (ours)** | 0.0140 | 0.0099 | 0.645 |
| 5 | SVR (RBF) | 0.0212 | 0.0099 | 0.643 |
| 6 | Random Forest | 0.0203 | 0.0118 | 0.574 |
| 7 | sklearn MLP | 0.0136 | 0.0149 | 0.462 |
| 8 | Gradient Boosting | 0.0276 | 0.0165 | 0.407 |
| 9 | QUANTECH MLP | 0.0204 | 0.0218 | 0.214 |
| 10 | VQC (Trained) | 0.0952 | 0.0581 | −1.089 |
| 11 | Quantum LSTM | 0.0537 | 0.0641 | −1.306 |

> Full per-model details, surface-level metrics, and timing data are in `benchmarks/results.csv` and `benchmarks/BENCHMARK_RESULTS.md`.

---

## Quick Start

### 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
pip install --upgrade pip
pip install torch merlinquantum
pip install -r requirements.txt
```

### 2. Train the pipeline

```bash
python src/train.py                # AE + quantum features + MLP head
python src/train_ridge.py          # QORC + Ridge readout (primary model)
```

| Phase | Description | Approx. time |
|-------|-------------|-------------|
| 1 | Sparse Denoising AE (200 epochs) | ~2 min |
| 2 | Encode all timesteps to latent codes | < 1 min |
| 3 | Ensemble QORC feature extraction | ~5–15 min |
| 4 | Ridge / MLP head training | < 1 min |

### 3. Generate predictions

```bash
python src/predict.py              # → outputs/predictions.xlsx
python src/predict_ridge.py        # → outputs/predictions_ridge.xlsx
```

### 4. Run benchmarks

```bash
python benchmarks/run_all.py               # all 11 models
python benchmarks/run_all.py --skip-slow   # skip VQC + Quantum LSTM
```

Results are saved to `benchmarks/results.csv` and `benchmarks/BENCHMARK_RESULTS.md`.

### 5. Generate figures

```bash
python benchmarks/create_figures.py        # → benchmarks/figures/*.png
```

---

## Project Structure

```
hackathon/
├── src/                          Core library
│   ├── preprocessing.py          Data loading, outlier treatment, normalization
│   ├── autoencoder.py            Sparse Denoising AE (224 → 20)
│   ├── quantum_reservoir.py      EnsembleQORC — 3 photonic reservoirs via MerLin
│   ├── hybrid_model.py           Temporal windowing, ClassicalHead MLP, loss
│   ├── ridge_model.py            QORC + Ridge readout (primary model)
│   ├── train.py                  Full training loop (AE → QORC → MLP)
│   ├── train_ridge.py            Ridge-only training (AE → QORC → Ridge)
│   ├── predict.py                Generate submission predictions
│   ├── predict_ridge.py          Ridge-based predictions
│   └── utils.py                  Config loading, seeding, device helpers
│
├── benchmarks/                   Reproducible model comparison
│   ├── run_all.py                Orchestrator — trains & evaluates 11 models
│   ├── data_utils.py             Shared data loading (train + test splits)
│   ├── metrics.py                Latent & surface-level evaluation metrics
│   ├── qorc_ridge.py             QORC + Ridge (our primary model)
│   ├── our_model.py              QUANTECH MLP evaluation
│   ├── classical_lstm.py         2-layer LSTM baseline
│   ├── quantum_lstm.py           VQC-enhanced LSTM
│   ├── random_forest.py          Multi-output Random Forest
│   ├── classical_ml.py           Ridge / Gradient Boosting / SVR / sklearn MLP
│   ├── vqc_model.py              Variational Quantum Circuit (MerLin)
│   ├── simple_pml_ridge.py       Single photonic unitary + Ridge
│   ├── create_figures.py         Generates all benchmark figures
│   ├── results.csv               Machine-readable results (val + test)
│   ├── BENCHMARK_RESULTS.md      Full Markdown report
│   └── figures/                  15 PNG visualizations
│
├── configs/
│   └── config.yaml               All hyperparameters
│
├── DATASETS/
│   ├── train.xlsx                494 × 224 swaption prices (training)
│   ├── test.xlsx                 6 × 224 held-out ground truth
│   └── test_template.xlsx        Submission template
│
├── outputs/                      Generated artifacts
│   ├── ae_weights.pt             Trained autoencoder weights
│   ├── head_weights.pt           MLP head weights
│   ├── ridge_model.joblib        Fitted Ridge readout
│   ├── quantum_features.npy      Pre-computed QORC features (1,215-dim)
│   ├── latent_codes.npy          AE-encoded latent codes (494 × 20)
│   ├── preprocessor.npz          Fitted scaler parameters
│   ├── predictions.xlsx          MLP-based submission
│   └── predictions_ridge.xlsx    Ridge-based submission
│
├── tests/                        Unit tests
│   ├── test_preprocessing.py
│   └── test_model_pipeline.py
│
├── website/                      Project showcase website
│   └── index.html
│
├── quantech.ipynb                Complete walkthrough notebook
├── requirements.txt              Python dependencies
├── PITCH.md                      Hackathon pitch deck notes
└── README.md
```

---

## Configuration

All hyperparameters live in `configs/config.yaml`:

```yaml
autoencoder:
  input_dim: 224
  hidden_dims: [128, 64]
  latent_dim: 20
  mask_ratio: 0.15           # denoising corruption ratio
  epochs: 200
  val_split: 50              # last N timesteps for validation

quantum_reservoir:
  ensemble:
    - { n_modes: 12, n_photons: 3, seed: 42 }   # 364 Fock features
    - { n_modes: 10, n_photons: 4, seed: 43 }   # 715 Fock features
    - { n_modes: 16, n_photons: 2, seed: 44 }   # 136 Fock features
  use_fock: true

hybrid_model:
  window_size: 5
  hidden_dims: [256, 128]
  dropout: 0.2
  epochs: 300

seed: 42
```

---

## Notebook

`quantech.ipynb` provides a self-contained, section-by-section walkthrough:

1. **Data Exploration** — load raw swaption surfaces, distribution analysis
2. **Preprocessing** — winsorization, robust scaling, normalization
3. **Autoencoder** — architecture, training, reconstruction quality
4. **Latent Space** — temporal dynamics, PCA/t-SNE visualization
5. **Quantum Reservoir** — photonic circuit design, Fock feature analysis
6. **Model Assembly** — windowing, feature concatenation, Ridge readout
7. **Test Evaluation** — walk-forward predictions on 6 held-out days
8. **Benchmarks** — comparative figures across all 11 models

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## References

- Li et al. (2025) — Quantum Reservoir Computing for realized volatility
- Notton et al. (2025) — Perceval Quest baselines
- Fujii & Nakajima (2017) — Foundational QRC theory
- [MerLin documentation](https://merlinquantum.ai)
- [Perceval documentation](https://perceval.quandela.net)

---

## License

This project was developed for the Q-volution Hackathon 2026 (Quandela Challenge).
