# Pitch: Hybrid Photonic QRC for Swaption Pricing

## Presentation Structure & Talking Points

---

## Slide 1 — Title & Hook

**Title:** Photonic Reservoir Computing Meets Financial Derivatives

**Hook:** "We use fixed photonic circuits as nonlinear feature extractors on a learned latent manifold of the swaption surface — combining the computational richness of quantum optics with the sample-efficiency of classical neural networks."

---

## Slide 2 — The Challenge

- **Input:** 494 daily swaption price surfaces, each with 224 tenor/maturity points
- **Tasks:**
  - Predict 6 future surfaces (autoregressive forecasting)
  - Impute 4-6 missing values in 2 partial surfaces
- **Constraint:** Must use MerLin (Quandela's photonic QML framework)

**Why it's hard:**
- 224-dimensional output space — far too large for any quantum circuit to handle directly
- Only 494 training samples — deep learning alone will overfit
- Swaption surfaces have strong structure (no-arbitrage constraints, smooth tenor/maturity dependence) that a naive model won't exploit

---

## Slide 3 — Architecture Overview

```
Raw Swaption Data (494 x 224)
  |
  v
[Robust Preprocessing] -- winsorize outliers, RobustScaler, MinMax to [0,1]
  |
  v
[Sparse Denoising Autoencoder] -- 224 -> 128 -> 64 -> 20 latent dims
  |                                learns a compact manifold of swaption surfaces
  v
[Temporal Windowing] -- 5-step sliding window + first-difference delta
  |                     captures recent dynamics and momentum
  v
[Ensemble QORC x3] -- 3 photonic reservoirs via MerLin (FIXED, not trained)
  |                    12-mode/3-photon + 10-mode/4-photon + 16-mode/2-photon
  |                    → 1,215 Fock-probability features
  v
[Classical MLP Head] -- quantum features (1215) + classical context (120)
  |                     → predicted next latent code (20 dims)
  v
[AE Decoder] -- 20 -> 64 -> 128 -> 224 swaption prices
```

---

## Slide 4 — Why This Architecture? (Justifications)

### Decision 1: Autoencoder for Dimensionality Reduction

**Why not PCA?**
- PCA is strictly linear — swaption surfaces have nonlinear dependencies (smile dynamics, skew, term structure curvature)
- A denoising AE learns a *nonlinear manifold* that captures these relationships
- Sparse L1 penalty forces each latent dimension to represent a distinct market factor (level, slope, curvature, smile)

**Why not feed raw 224 dims to the quantum circuit?**
- Quantum simulation scales exponentially with modes/photons — 224 modes is computationally impossible
- 20 latent dimensions are compact enough for efficient phase encoding

**Why denoising?**
- Random masking during training (15% of features zeroed) teaches the AE to reconstruct from partial information
- This directly enables Task B (missing data imputation) — the AE naturally fills gaps

### Decision 2: Ensemble of 3 QORC Reservoirs

**Why ensemble?**
- Different (n_modes, n_photons) configurations explore different regions of Fock space
- Diversity: 12/3 captures cubic interactions, 10/4 captures quartic, 16/2 captures pairwise
- Total 1,215 quantum features — rich enough for the MLP head, but computed once (no training needed)

**Why fixed (not trained) reservoirs?**
- Reservoir computing paradigm: random fixed nonlinear expansion + trained linear readout
- With only 494 samples, training circuit parameters would overfit immediately
- Fixed reservoirs are computationally cheap (simulate once, cache features)
- Follows foundational QRC theory (Fujii & Nakajima, 2017)

**Why Fock-space probabilities as features?**
- Fock-state probabilities are inherently nonlinear functions of input phases
- They capture multi-photon interference effects — a computational resource unique to photonic systems
- C(n+m-1, n) output dimensions naturally create a high-dimensional feature space

### Decision 3: Sandwich Circuit (U | PS | U)

**Why this circuit structure?**
- Haar-random unitaries provide maximal mode-mixing (every photon path interferes with every other)
- Phase shifters in the middle encode input data — each latent dimension maps to one phase
- Second unitary re-mixes the encoded photons before measurement
- This is the standard QORC architecture from Li et al. (2025) and reproduced_papers

### Decision 4: Phase Encoding via Sigmoid → [0, 2π]

**Why sigmoid?**
- Maps arbitrary real-valued latent features to bounded [0, 2π] phase range
- Smooth, differentiable — gradients flow through during head training
- Full 2π range exploits the complete periodicity of optical phase

**Why not amplitude encoding?**
- Challenge constraints prohibit amplitude encoding
- Phase encoding is natural for photonic circuits (physically corresponds to optical path length)

### Decision 5: Predict in Latent Space, Not Price Space

**Why predict z_{t+1} instead of prices directly?**
- 20-dim prediction is much easier than 224-dim prediction
- Latent space is smoother and lower-noise (AE acts as a denoiser)
- The AE decoder ensures predictions respect the learned surface manifold — no arbitrage-violating outputs

### Decision 6: Combined Latent + Surface Loss

**Loss = MSE(z_pred, z_true) + 0.1 * MSE(surface_pred, surface_true)**

- Latent loss provides smooth, direct gradients early in training
- Surface loss provides end-to-end accuracy signal — penalizes reconstructions that diverge in price space
- The 0.1 weight prevents surface loss from dominating (it's noisier, higher-dimensional)
- Gradients flow through the frozen AE decoder back to the head via chain rule

### Decision 7: Robust Preprocessing (Winsorize + RobustScaler + MinMax)

**Why not just StandardScaler?**
- Financial data has fat tails and outliers — a single extreme value can distort the mean/std
- Winsorizing clips to 1st/99th percentile — removes extreme outliers without losing samples
- RobustScaler uses median/IQR — resistant to remaining outliers
- MinMax to [0,1] — required for Sigmoid decoder output and well-behaved AE training
- Training bounds are saved and applied identically at test time — no data leakage

---

## Slide 5 — Quantum Advantage Argument

**What the quantum reservoir provides that a classical system doesn't:**

1. **Exponential feature space:** 3 photons in 12 modes → 364 Fock outcomes. The photonic circuit computes all C(n+m-1,n) interference probabilities simultaneously — this would require an exponentially-sized classical neural network to replicate.

2. **Multi-photon interference:** Each Fock probability depends on the permanent of a submatrix of the unitary — computing matrix permanents is #P-hard classically. The photonic circuit evaluates these for free.

3. **Natural nonlinearity:** Unlike a random kitchen-sink or RBF feature expansion, quantum features have a physically-motivated nonlinear structure that captures genuine multi-body correlations.

4. **Ensemble diversity:** Different photon numbers create qualitatively different feature spaces (quadratic, cubic, quartic correlations) — not just different random seeds of the same function class.

---

## Slide 6 — Training Pipeline

| Phase | What | Time (RTX 4090) |
|-------|------|-----------------|
| 1 | Train Sparse Denoising AE (200 epochs) | ~2 min |
| 2 | Encode all 494 timesteps to 20-dim latent codes | <1 min |
| 3 | Run ensemble QORC on all 489 windows (cached) | ~5-15 min |
| 4 | Train Classical MLP Head (300 epochs) | ~3 min |

**Key efficiency insight:** Quantum features are pre-computed once in Phase 3. Since the reservoirs are fixed, we don't need to re-run photonic simulation every epoch. This makes training fast despite the quantum component.

---

## Slide 7 — Prediction Strategy

### Task A: Future Prediction (6 rows, all values unknown)

- Seed with last 5 known latent codes from training
- Predict z_{t+1} using QORC + head
- Decode z_{t+1} → 224 prices via AE decoder
- Shift window forward, repeat (autoregressive rollout)
- Each step uses the model's own predictions as context

### Task B: Missing Data Imputation (2 rows, 4-6 missing values)

- Fill NaN positions with training column medians (reasonable initial guess)
- Normalize and pass through the denoising AE (encode → decode)
- The AE reconstructs the full surface, exploiting learned tenor/maturity correlations
- Keep original observed values; only replace NaN positions with AE output
- This leverages the AE's denoising training — it was specifically trained to reconstruct from masked inputs

---

## Slide 8 — Assumptions & Limitations

### Assumptions
1. **Stationarity:** The latent dynamics learned from 494 historical surfaces generalize to the prediction horizon. If the market regime shifts, autoregressive predictions may diverge.
2. **Manifold hypothesis:** Swaption surfaces live on a ~20-dimensional nonlinear manifold. This is well-supported by financial literature (level, slope, curvature, smile explain most variation).
3. **Reservoir expressivity:** Fixed random photonic circuits provide sufficiently rich features without training. Justified by QRC theory: random reservoirs work well when the input signal has temporal structure.
4. **Phase encoding sufficiency:** Encoding latent features as phases (not amplitudes) captures enough information. This is a constraint of the challenge, not a choice.

### Limitations
1. **Error compounding:** Autoregressive rollout for 6 steps means prediction quality degrades with each step. No uncertainty quantification is provided.
2. **Small dataset:** 494 samples → AE may overfit despite regularization (denoising + L1 sparsity + early stopping). The 170:1 parameter-to-sample ratio is aggressive.
3. **No no-arbitrage constraints:** The model doesn't enforce financial constraints (e.g., positive prices, convexity). The AE learns these implicitly from data but doesn't guarantee them.
4. **Simulation vs. QPU:** Results use photonic simulation, not real quantum hardware. QPU execution would introduce shot noise and hardware errors.

---

## Slide 9 — Results

*(Fill in after training)*

### Metrics to show:
- AE reconstruction RMSE (normalized and raw price scale)
- Hybrid model validation RMSE (latent space and price space)
- Quantum vs. classical baseline comparison (same pipeline without QORC)
- Predicted vs. actual surfaces (heatmap visualization)
- Latent code trajectories (time series of 20 dimensions)
- Training curves (loss vs. epoch for both AE and head)

### Visualizations:
1. **Swaption surface heatmaps** — tenor x maturity grid, predicted vs. ground truth
2. **Latent space evolution** — 20-dim latent codes over 494 timesteps (shows smooth dynamics)
3. **Training curves** — convergence plots for AE and head phases
4. **Missing data reconstruction** — before/after imputation side-by-side

---

## Slide 10 — Novelty & Creativity

1. **Latent-space QRC:** Instead of applying quantum features to raw high-dimensional data (impossible) or PCA projections (too linear), we apply QORC to autoencoder latent codes — a novel combination that bridges quantum and deep learning.

2. **Ensemble diversity through physics:** The three reservoirs use different photon numbers, creating qualitatively different nonlinear feature spaces (pairwise, cubic, quartic photon correlations). This is richer than simply using different random seeds.

3. **Denoising AE as dual-purpose module:** The same AE handles both dimensionality reduction (for the QRC pipeline) AND missing data imputation (for Task B). Training with random masking creates a model that naturally fills gaps.

4. **Pre-computed quantum features:** By separating the fixed quantum computation from trainable classical optimization, we avoid the barren plateau problem entirely and make training efficient.

5. **Combined loss with decoder gradient flow:** The surface loss term propagates gradients through the frozen decoder's Jacobian back to the head, providing an end-to-end training signal without requiring decoder parameter updates.

---

## Slide 11 — Conclusion

- **Hybrid architecture** combines quantum computing's natural nonlinearity with classical neural networks' sample efficiency
- **Practical design** — trains in ~20 minutes on consumer GPU, quantum features computed once
- **Principled choices** — every architectural decision is justified by the data constraints (small N, high-dimensional output, missing values)
- **Extensible** — can swap in QPU execution, add more reservoirs, or increase latent dim without architectural changes

---

## Appendix — Key Numbers

| Component | Value |
|-----------|-------|
| Training samples | 494 |
| Price columns | 224 (14 tenors x 16 maturities) |
| AE latent dim | 20 |
| AE parameters | ~76,000 |
| Window size | 5 timesteps |
| Classical context dim | 120 (20 * 6) |
| Quantum features | 1,215 (364 + 715 + 136) |
| Head input dim | 1,335 (1215 + 120) |
| Head parameters | ~380,000 |
| Reservoir configs | 12/3, 10/4, 16/2 modes/photons |
| Phase encoding range | [0, 2π] |
| Total training time | ~20 min (RTX 4090) |
