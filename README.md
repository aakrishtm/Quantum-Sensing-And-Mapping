# GraviQ: Simulating Subsurface Mapping with a Qubit-Based Gravimeter

Our project simulates how a cold-atom–based quantum gravimeter could be used for subsurface mapping entirely in software. We begin by constructing a simplified subsurface model in which we assign densities to points in a 1D line or 2D grid, introducing features such as low-density voids or high-density ore regions.

## Quickstart

```bash
git clone <repo-url>
cd GraviQ-quantum-sensing
make install
make data    # Generate full training data (density_grid + tunnel_mask + metadata)
make train   # Train the model
make infer   # Run inference and produce example plot
```

- **`make install`** — installs the package in editable mode (`pip install -e .`).
- **`make data`** — generates training data (500 samples with density grids, tunnel masks, metadata).
- **`make train`** — trains the model; saves `checkpoints/best_model.pth`.
- **`make infer`** — runs inference (needs trained model and `training_data/`).

To run the web demo: `make run-app` or `python app.py`.

**If port 5000 is already in use:** run on another port, e.g. `PORT=5001 make run-app` or `PORT=5001 python app.py`.

## Development

- **Train:** `make train` (calls `scripts/train.py`).
- **Test (placeholder):** `make test` — currently just checks `import graviq`.

**Evaluation:** Inference reports "Classification Accuracy" (tunnel vs no-tunnel) and Dice/IoU. Very high or 100% classification accuracy on this synthetic data is common and can indicate overfitting; focus on Dice and IoU for segmentation quality.

## Package layout

- `src/graviq/` — main package: `data/`, `models/`, `physics/`, `sim/`, `viz/`, training and inference logic.
- `scripts/` — thin entrypoints: `train.py`, `infer.py`, `app.py`.

## Forward models

The ML pipeline can optionally feed the model **interferometer readout** (normalized signal in [0, 1]) instead of the raw physical field. When the interferometer config is enabled in the dataset, the input is computed as: physical grid → phase shift → (optional wrap) → phase-to-signal → sensor noise. The model then sees a realistic readout rather than raw density or acceleration.
