# Q-volution Hackathon 2026 - Quandela Challenge

## Project: Hybrid Photonic QRC for Swaption Pricing

### Overview
Quantum Machine Learning model using MerLin (Quandela's photonic QML framework) to predict swaption prices. Combines Quantum Optical Reservoir Computing (QORC) with classical neural networks.

### Architecture
```
Raw Swaption Data (494×224)
  → Robust Preprocessing (winsorize, RobustScaler, MinMax)
  → Sparse Denoising Autoencoder (224 → 20 latent dims)
  → Temporal Windowing (window=5)
  → Ensemble QORC (3 photonic reservoirs, MerLin)
  → Classical MLP Head (quantum + classical features → latent prediction)
  → AE Decoder (20 → 224 swaption prices)
```

### Tech Stack
- **Quantum:** MerLin (`pip install merlinquantum`), built on Perceval
- **ML:** PyTorch, scikit-learn
- **Data:** pandas, openpyxl
- **Hardware constraints:** Sim ≤20 modes, ≤10 photons | QPU ≤24 modes, ≤12 photons

### Project Structure
```
hackathon/
├── src/
│   ├── preprocessing.py       # Data loading, outliers, normalization
│   ├── autoencoder.py         # Sparse Denoising Autoencoder
│   ├── quantum_reservoir.py   # QORC ensemble (MerLin)
│   ├── hybrid_model.py        # Full pipeline
│   ├── train.py               # Training loop
│   ├── predict.py             # Test predictions
│   └── utils.py               # Helpers
├── configs/config.yaml        # Hyperparameters
├── DATASETS/                  # train.xlsx, test_template.xlsx, sample
├── outputs/                   # Checkpoints, predictions
├── PLAN.md                    # Detailed implementation plan
└── CLAUDE.md                  # This file
```

### Datasets
- `train.xlsx`: 494 rows × 225 cols (Date + 224 tenor/maturity prices). Clean, no missing values.
- `test_template.xlsx`: 8 rows × 226 cols (Date + Type + 224 cols with NaNs). Types: "Future prediction", "Missing data"
- `sample_Simulated_Swaption_Price.xlsx`: Reference output format

### Key MerLin Patterns
- `ML.QuantumLayer(circuit=..., input_parameters=["px"], trainable_parameters=[], ...)` for fixed reservoir
- `ML.MeasurementStrategy.probs(computation_space=ML.ComputationSpace.FOCK)` for Fock probabilities
- `pcvl.Matrix.random_unitary(n_modes)` for Haar-random interferometer
- Circuit structure: `Unitary // PhaseShifters // Unitary` (sandwich)

### Judging Criteria
- Accuracy/Technical Merit: 35%
- Creativity/Novelty: 35%
- Presentation/Documentation: 30%

### Key References
- ReservoirQC.pdf - QRC for realized volatility (Li et al., 2025)
- PhotonicML.pdf - Perceval Quest baselines (Notton et al., 2025)
- OriginReservoirQ.pdf - Foundational QRC (Fujii & Nakajima, 2017)
- MerLin reproduced_papers: QORC, qrc_memristor implementations

### User Setup
- RTX 4090, 64GB RAM, Core i9
- Strong background in both quantum computing and ML
