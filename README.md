## GraviQ: Quantum Gravimetry for Subsurface Mapping

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange.svg)
![Qiskit](https://img.shields.io/badge/qiskit-aer_gpu-8A2BE2.svg)
![CUDA](https://img.shields.io/badge/CUDA-H100--ready-76B900.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

GraviQ is an ongoing, months-long independent research project in **quantum gravimetry** and **inverse problems**.  

The end goal is to prototype an end-to-end pipeline that simulates a cold-atom–based quantum gravimeter, generates synthetic gravity-gradient data with realistic quantum noise, and trains deep models to recover subsurface structure (voids, tunnels, ore bodies) from those measurements.

---

### Background

Conventional gravimetry workflows use classical forward models and deterministic noise; they struggle to capture the statistics of **shot noise**, **phase decoherence**, and **control errors** that dominate real quantum sensors.  
GraviQ treats the entire stack as software:

- Synthetic **3D density models** of the subsurface (rock, void, ore)  
- Quantum forward model of a **cold-atom interferometer** measuring the vertical gravity gradient $G_{zz}$  
- A U-Net–style inverse model that infers **subsurface tunnels and anomalies** from 2D gravity slices

The project is intentionally reproducible, configurable, and designed for rapid iteration on physics and ML components.

---

### Project Overview

**Problem statement.** Given noisy 2D measurements of the gravity gradient $G_{zz}(x, z)$ at the surface, reconstruct a discretized 3D density field $\rho(x, y, z)$ capturing:

- low-density **voids** and tunnels  
- high-density **ore** or mineralization  
- background rock with spatially varying density

In the current implementation, GraviQ solves a reduced but representative version of this inverse problem:

- **Forward direction**:  
  - Generate 2D density cross-sections with stochastic tunnels and ore bodies.  
  - Simulate a **quantum cold-atom interferometer** sampling an effective $G_{zz}$ field over a $60 \times 150$ grid.  
  - Inject realistic noise (phase damping, Gaussian readout noise) to approximate hardware behaviour on an H100-class GPU.

- **Inverse direction**:  
  - Train a **U-Net** on Gzz-like maps to perform **semantic segmentation** of tunnels vs non-tunnel regions.  
  - Use Dice/IoU metrics to evaluate how well the model recovers the tunnel mask under domain shift and noise.

The longer-term trajectory is a full 3D voxelized inverse problem (see “Active Development”).

---

### Technical Stack

- **Quantum simulation layer**
  - `qiskit` + `qiskit-aer-gpu`:  
    - `AerSimulator(method="density_matrix", device="GPU")` for density-matrix simulation with **phase damping** noise.  
    - Custom Ramsey interferometer circuits with per-pixel dephasing channels to emulate **quantum shot noise** and **decoherence**.  
  - **Cirq (conceptual / experimental)**: the architecture is designed to admit alternative backends (e.g., Cirq-based parameterized circuits) for benchmarking different simulators.

- **Machine learning layer**
  - **PyTorch** 2.x  
    - Lightweight U-Net (`src/graviq/models/unet.py`) with **BatchNorm** and **spatial Dropout2d** for regularization.  
    - Custom loss: **BCEWithLogits + Dice** with **pos-weighting** and **label smoothing** for stability on highly imbalanced masks.  
  - Training orchestration (`src/graviq/training.py`)  
    - Gradient-stable combined loss with tunable weights.  
    - `ReduceLROnPlateau` scheduler for automatic learning-rate annealing.

- **Simulation + physics**
  - Density and tunnel generator: `density_grid_generator.py`  
  - Quantum Gzz generator: `generate_gzz_grids.py` (Qiskit Aer GPU, batched circuits with phase damping)  
  - Interferometer model: `src/graviq/physics/interferometer.py` (Mach–Zehnder–style phase shift, wrapping, and readout signal).

- **Infrastructure**
  - CUDA/H100-focused configuration (via `qiskit-aer-gpu` and PyTorch CUDA backends).  
  - Pure Python + NumPy/PyTorch, kept portable enough to run on CPU-only CI by design (with graceful GPU fallbacks).

---

### Core Pipeline

#### 1. Synthetic subsurface generation

Density grids and labels are created by `density_grid_generator.py`:

- The subsurface is discretized into a $60 \times 150$ grid with rock, void, and ore densities.  
- Tunnels are drawn as curved low-density paths with variable length, thickness, and curvature.  
- Each sample produces:
  - `density_grid_XXX.npy` — density field  
  - `tunnel_mask_XXX.npy` — ground-truth segmentation mask  
  - `metadata_XXX.json` — path geometry, bounding boxes, and tunnel statistics

#### 2. Quantum Gzz simulation (Qiskit Aer GPU)

`generate_gzz_grids.py` replaces classical approximations with a **Ramsey interferometer** on a Qiskit Aer density-matrix backend:

- For each quantized density level $\rho$:
  - Build a single-qubit Ramsey circuit:  
    $ H \rightarrow \text{delay}(T) \rightarrow H \rightarrow \text{measure} $
  - Construct a phase-damping channel with probability  
    $ p(\rho, T) = 1 - e^{-\rho T} $  
  - Attach the channel as noise (`phase_damping_error`) via a `NoiseModel`.
- Group pixels by rounded density to **batch circuits** and maximize GPU occupancy:

```python
from qiskit_aer import AerSimulator, AerError
from qiskit_aer.noise import NoiseModel, phase_damping_error

sim = AerSimulator(method="density_matrix", device="GPU")  # falls back to CPU if GPU unavailable

circuits = [ramsey_circuit(delay_ns) for _ in range(batch_size)]
noise_model = NoiseModel()
noise_model.add_quantum_error(phase_damping_error(p), ["id", "delay"], [0])

result = sim.run(circuits, noise_model=noise_model, shots=shots).result()
gzz_values = [
    (counts.get("0", 0) - counts.get("1", 0)) / shots
    for counts in (result.get_counts(i) for i in range(batch_size))
]
```

- Output: `gzz_grid_XXX.npy` (shape $60 \times 150$, `float32`) stays **API-compatible** with the fast approximation so downstream PyTorch code is unchanged.
- Runtime robustness: if `qiskit-aer-gpu` or CUDA drivers are missing, the generator **automatically falls back** to CPU (`device="CPU"`), keeping CI green.

#### 3. Data pipeline and augmentation

The training dataset (`src/graviq/data/dataset.py`) is centered on `TunnelDataset`:

- Loads `gzz_grid_*.npy` and `tunnel_mask_*.npy` pairs.  
- Optional **interferometer** and **sensor noise** stages:
  - `apply_interferometer_model` — phase shift $ \Delta \phi = k_{\text{eff}} a T^2 $, optional wrapping to $[- \pi, \pi]$, then signal mapping.  
  - `apply_sensor_model` — Gaussian noise and blur, with deterministic per-sample seeding.
- Preprocessing matches the deployed Flask app:
  - Flip sign so tunnels become high-response peaks.  
  - Per-sample min-max normalization to $[0, 1]$.  
  - Channel-first tensors `(1, H, W)` are returned.

On top of this, the dataloader applies **on-the-fly spatial augmentation**:

- Random horizontal / vertical flips for the training split only.  
- Validation split remains untouched for honest evaluation.

```python
from graviq.data import get_dataloaders

train_loader, val_loader = get_dataloaders(
    data_dir="training_data",
    batch_size=8,
    train_split=0.8,
    num_workers=0,  # can be increased on larger machines
)
```

#### 4. U-Net architecture and regularization

The inverse model is a compact U-Net (`src/graviq/models/unet.py`):

- Encoder–decoder with skip connections and feature sizes `[32, 64, 128, 256]`.  
- Each block uses **Conv2d → BatchNorm2d → ReLU** × 2, followed by **Dropout2d**:

```python
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout: float = 0.1):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        layers.append(nn.Dropout2d(p=dropout))
        self.double_conv = nn.Sequential(*layers)
```

This **spatial dropout** reduces overfitting on limited synthetic data while preserving spatial structure.

#### 5. Training loop and loss design

Training logic lives in `src/graviq/training.py`:

- **Loss**: combined BCEWithLogits + Dice with:
  - `pos_weight` computed from the training set (tunnel = minority class).  
  - Tunable `bce_weight` / `dice_weight` (default 0.3 / 0.7).  
  - **Binary label smoothing** (default $\epsilon = 0.05$):  
    targets are nudged toward 0.5 to dampen gradient spikes and hard overconfidence.

- **Scheduler**: `ReduceLROnPlateau` on validation loss to automatically shrink the learning rate when the valley stalls.

- **Metrics**: Dice, IoU, precision, recall; best model checkpoint is chosen by validation Dice.

```python
from graviq.training import train

train(
    data_dir="training_data",
    num_epochs=100,
    batch_size=8,
    learning_rate=1e-3,
)
```

The combination of spatial dropout, augmentation, pos-weighted BCE, and label smoothing is designed to **stabilize training under domain shift** when the quantum forward model changes (e.g., different shot counts, decoherence rates, or noise models).

---

### Current Research Challenges

The active research focus is on **robustness under realistic quantum noise** and **generalization across forward models**:

- **Domain shift under quantum shot noise / decoherence**
  - Moving from analytic Gzz approximations to Qiskit-based density-matrix simulations changes both the marginal distribution and spatial correlations of the input.  
  - U-Net must stay performant as `shots`, `t_evolution`, and dephasing parameters vary, and as the effective SNR collapses in certain regions.

- **Overfitting and gradient instability**
  - Tunnel pixels occupy a small fraction of the grid; naive BCE collapses to predicting “no tunnel”.  
  - High-confidence misclassifications can cause sharp loss spikes when new quantum noise settings are introduced.
  - The current solution combines:
    - pos-weighted BCEWithLogits  
    - Dice loss emphasizing overlap  
    - label smoothing for softer gradients  
    - spatial dropout and random flips for regularization

- **Data pipeline throughput**
  - Gzz maps are generated via batch-compiled Qiskit circuits to keep GPU utilization high and RAM stable.  
  - PyTorch dataloaders stream `.npy` grids on demand; pinning memory and enabling workers can be tuned per target environment.

---

### Active Development Roadmap

1. **3D volumetric inverse model**
   - Extend from 2D cross-sections to full **3D voxel volumes** with 3D convolutions and skip connections.  
   - Target resolutions of $60 \times 150 \times 60$ (>$5.4 \times 10^5$ voxels) with memory-aware tiling on H100-class GPUs.

2. **Parameterized quantum circuits for 3D forward models**
   - Replace per-pixel Ramsey circuits with **parameterized quantum circuits** that share structure across voxels.  
   - Exploit Qiskit Aer GPU batching and (optionally) Cirq-based parameter sweeps to maximize throughput and reduce compilation overhead for large 3D grids.

3. **Full gravity-gradient tensor and multi-class segmentation**
   - Upgrade from scalar $G_{zz}$ to the full gravity-gradient tensor $G_{ij}$ (e.g., $G_{xy}, G_{xz}, G_{yz}$).  
   - Extend labels from binary (tunnel vs background) to **multi-class segmentation** (e.g., void, standard rock, high-density ore, rare earth deposits).  
   - Investigate cross-component consistency losses that couple predictions across different tensor components.

---

### Quickstart

```bash
git clone <repo-url>
cd GraviQ-quantum-sensing

make install
make data    # Generate density_grid + tunnel_mask + Gzz grids
make train   # Train the U-Net inverse model
make run-app # Launch the Flask demo dashboard
```

- **`make install`** — install the package in editable mode (`pip install -e .`).  
- **`make data`** — run density + tunnel synthesis and quantum Gzz generation (Qiskit Aer).  
- **`make train`** — train the U-Net and write checkpoints to `checkpoints/`.  
- **`make run-app`** — start a Flask app for interactive visualization and what-if analysis.

If port 5000 is taken, run e.g. `PORT=5001 make run-app`.

**GPU vs CPU:**  
On a CUDA/H100 machine with `qiskit-aer-gpu` installed, Gzz generation uses GPU density-matrix simulation by default.  
On CPU-only hosts or CI, the code automatically falls back to CPU simulation; you can also force this via `--device CPU` on `generate_gzz_grids.py`.

---

### Example: Inspecting a Single Sample

```python
import numpy as np
import torch
from graviq.data import get_dataloaders

train_loader, _ = get_dataloaders("training_data", batch_size=4)
batch = next(iter(train_loader))

gzz = batch["input"]      # (B, 1, 60, 150), preprocessed Gzz-like input
mask = batch["mask"]      # (B, 1, 60, 150), ground-truth tunnel mask
print(gzz.shape, mask.shape, gzz.min().item(), gzz.max().item())
```

This provides a compact starting point for experimenting with quantum gravimetry inverse problems, alternative simulators, and robust deep inverse models. The codebase is intentionally modular so both the **quantum forward model** and the **ML inverse model** can be swapped or extended with minimal friction.
