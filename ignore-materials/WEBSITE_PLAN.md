# Website Redesign Plan

Redesign the QuanTech website (`website/index.html`) and update `benchmarks/create_figures.py` to generate richer, more insightful figures. Study the QEDI team's approach at `https://github.com/HUmutI/EPFL-Quandela-Qedi/tree/main/qedi_website` for inspiration — they won a previous edition of this hackathon.

## What QEDI Does Better Than Us (Fix These)

1. **Interactive data visualizations** — They export model results to JSON and render interactive Plotly.js charts (hoverable, zoomable). We only have static matplotlib PNGs. Add Plotly.js charts.
2. **Animated photonic circuit** — They have a live HTML5 Canvas animation showing photons traveling through optical modes and interferometers inside their architecture diagram. We have nothing animated in our pipeline section.
3. **Animated challenge visualization** — They show the two tasks (forecasting + imputation) as animated CSS grids: one has future columns fading in, the other has missing cells blinking. We just have static text cards.
4. **Particle hero background** — They have a canvas-based particle network animation (photon-like). We just have a CSS grid animation — upgrade to a canvas particle system with purple/violet photon dots and quantum-inspired connection lines.
5. **The pipeline diagram should be a dark horizontal strip** — Their architecture section uses a dark background (`#0f172a`) horizontal pipeline with embedded live circuit animation. Our vertical accordion cards are functional but visually bland compared to theirs.
6. **Data-driven results section** — They embed actual prediction data as JSON and show interactive plots of actual vs predicted prices. We should do the same — show our model's predictions against ground truth.

## Part 1: New Figures via `create_figures.py`

Update `benchmarks/create_figures.py` to generate these additional figures (keep all 8 existing ones, add new ones):

### Fig 9: Swaption Surface Heatmaps (3-panel)
- Load the actual training data from `DATASETS/train.xlsx` using `src/preprocessing.py`
- Show 3 heatmaps side-by-side: first training day, middle training day, last training day
- X-axis: Maturity (16 values), Y-axis: Tenor (14 values)
- Use `viridis` colormap, shared colorbar
- Title: "Swaption Surface Evolution Over Time"

### Fig 10: AE Reconstruction Quality
- Load `outputs/ae_weights.pt` and `outputs/preprocessor.npz`
- Reconstruct the full training set through the AE
- Panel 1: Scatter plot of original vs reconstructed values (subsample for speed)
- Panel 2: Reconstruction error heatmap (14x16 tenor x maturity grid, averaged over all timesteps)
- Shows where the AE is accurate and where it struggles

### Fig 11: Latent Code Trajectories
- Load `outputs/latent_codes.npy` (494x20)
- Plot first 5 latent dimensions as time series over 494 timesteps
- Each dim in a different color, subtle alpha
- Shows the smooth temporal dynamics the model learns

### Fig 12: Quantum Feature Distribution Comparison
- Load `outputs/quantum_features.npy`
- 3-panel histogram: one per reservoir (features 0-363 = R1, 364-1078 = R2, 1079-1214 = R3)
- Shows the different statistical properties of each reservoir's Fock features
- Helps visualize why ensemble diversity matters

### Fig 13: Prediction vs Ground Truth (Validation)
- Run the QORC+Ridge model on validation data
- Load `outputs/latent_codes.npy`, use last 50 as validation
- Use the ridge model from `outputs/ridge_model.joblib` (or `benchmarks/qorc_ridge.py` logic)
- Panel 1: 3 sample swaption columns (pick representative tenors) — predicted vs actual over validation timesteps
- Panel 2: Prediction error distribution (histogram)
- This is the most impactful figure — shows our model actually works

### Fig 14: Quantum Feature Importance
- Load the fitted Ridge model
- Extract Ridge coefficients (20 output dims x 1335 input features)
- Show average absolute coefficient for quantum features vs classical features
- Bar chart: "Quantum Features" vs "Classical Features" average importance
- If quantum features have meaningful weight, this proves they carry information beyond classical context

### Export JSON for Plotly.js
Also export these JSON files to `website/assets/`:
- `surface_data.json` — 3 sample swaption surfaces (first, middle, last training day) with tenor/maturity labels
- `latent_trajectories.json` — first 5 latent dims over 494 timesteps
- `prediction_comparison.json` — validation set predictions vs actuals for 3 representative price columns
- `benchmark_results.json` — the full results.csv data in JSON format for interactive table rendering

## Part 2: Website Redesign

Update `website/index.html` with these changes. Keep the existing structure/content but enhance with:

### Hero Section
- Replace CSS grid animation with an HTML5 Canvas particle network:
  - 60-80 particles, 20% colored violet-500, rest gray-300 with low opacity
  - Particles drift slowly, bounce off edges
  - Draw semi-transparent purple connection lines between particles closer than 120px
  - Creates a "quantum network" visual similar to QEDI but in our purple palette
  - Use `requestAnimationFrame` for smooth 60fps animation

### Challenge Section
- Replace the 3 static text cards with animated task visualizations:
  - **Task A card:** A 7x7 CSS grid where the rightmost 2 columns cycle between empty/gray and filled/violet every 3 seconds (simulating future prediction appearing)
  - **Task B card:** A 7x7 CSS grid where 3-4 random cells blink between gray (missing) and green (imputed) every 2 seconds
  - Keep the constraint card as-is but keep visual consistency

### Architecture Section
- Convert the vertical accordion pipeline into a **dark horizontal strip** (background `#1e1b4b`):
  - Each pipeline node is a compact card with icon + name + key metric
  - Horizontal arrows between nodes with a pulsing `flow-right` CSS animation
  - The QORC node (step 5) gets a special glow border and an **embedded mini canvas** showing a simplified photonic circuit animation:
    - Draw 6-8 horizontal mode lines in blue/purple
    - Draw 2 rectangular interferometer blocks
    - Animate 2-3 glowing dots (photons) traveling along the mode lines, switching at interferometers
    - Small canvas (300x150px), runs at 30fps
  - Clicking any node opens detail text below (keep accordion behavior but horizontal layout)
  - On mobile: stack vertically with connecting arrows

### Results Section — Add Interactive Plotly Charts
- Add Plotly.js via CDN: `<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>`
- **Interactive Swaption Surface**: 3D surface plot of a sample swaption surface (from `surface_data.json`). Rotatable, hoverable. Use Plasma colorscale.
- **Interactive Latent Trajectories**: Line chart of 5 latent dims over time (from `latent_trajectories.json`). Hover to see values. Toggle dims on/off via legend.
- **Interactive Prediction Comparison**: Overlaid line charts of actual vs predicted for 3 representative price columns (from `prediction_comparison.json`). Shows the model tracking real prices.
- Keep all existing static figure images as well — they're good for offline viewing and presentation screenshots.

### Benchmarks Section — Interactive Table
- Replace the hardcoded HTML table with a Plotly.js table rendered from `benchmark_results.json`
- Our model row highlighted in violet
- Sortable columns (Plotly tables support this natively)
- Keep the existing static figures below

### New Section: "Quantum vs Classical — The Evidence"
Add this section between Results and Benchmarks. Purpose: make the quantum advantage argument data-driven. Include:
- The **Feature Importance** figure (Fig 14) — shows Ridge gives meaningful weight to quantum features
- A side-by-side metric card: "With Quantum Features: MSE 0.014" vs "Without (Ridge only): MSE 0.011" — show the numbers honestly, but frame it as "quantum features provide complementary information to classical context, achieving competitive performance with inherently richer feature representations"
- The **Quantum Feature Distribution** figure (Fig 12) — shows the three reservoirs produce different statistical distributions

### Image Lightbox Enhancement
- Add a caption overlay at the bottom of the lightbox modal (show the figure's `.caption` text)
- Add left/right keyboard navigation between figures (arrow keys)
- Add a subtle dark vignette border in the lightbox

### Footer
- Add team member names as clickable links (leave as placeholders: "Member 1", "Member 2", etc.)
- Add "Built with MerLin + Perceval" with small text

## Technical Requirements
- Single HTML file, keep Tailwind CDN + add Plotly.js CDN
- Canvas animations should be performant — requestAnimationFrame, no memory leaks, cleanup on page unload
- Plotly charts should be responsive (use `Plotly.newPlot(el, data, layout, {responsive: true})`)
- JSON files loaded via fetch() — handle loading states gracefully (show "Loading..." placeholder)
- All canvas animations should pause when not in viewport (use IntersectionObserver) to save CPU
- Keep smooth scroll, reveal animations, mobile responsiveness from existing site
- No external dependencies beyond Tailwind CDN, Google Fonts (Inter + Space Grotesk), and Plotly.js CDN

## File Structure After Changes
```
website/
├── index.html          # Updated single-page site
├── assets/
│   ├── surface_data.json
│   ├── latent_trajectories.json
│   ├── prediction_comparison.json
│   └── benchmark_results.json
└── index_dark_backup.html  # Keep existing backup
benchmarks/
├── create_figures.py   # Updated with 6 new figures + JSON export
├── figures/            # All PNGs (existing 8 + new 6)
└── ...                 # Existing benchmark files unchanged
```

## Run Order
1. First run `python benchmarks/create_figures.py` — generates all figures + JSON assets
2. Then open `website/index.html` in a browser — it loads JSON via fetch and renders Plotly charts
