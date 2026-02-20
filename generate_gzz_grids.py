"""Fast vectorized Gzz generation. No Qiskit—uses analytical model + binomial shot noise."""

import argparse
import os

import numpy as np

from graviq.sim import gzz_approximation


def density_to_gzz_grid(
    density_grid: np.ndarray,
    t_evolution: float = 30e-6,
    shots: int = 500,
) -> np.ndarray:
    """
    Vectorized Gzz with shot noise. Same output format as Qiskit version.
    ~milliseconds instead of minutes per sample.
    """
    density_grid = np.asarray(density_grid, dtype=np.float32)
    # Analytical Gzz (matches gzz_approximation)
    gzz_exact = gzz_approximation(density_grid, t_evolution=t_evolution).astype(np.float64)
    # P0 = (1 + Gzz) / 2
    p0 = np.clip((1.0 + gzz_exact) / 2.0, 0.0, 1.0)
    # Shot noise: sample binomial for all pixels at once
    counts_0 = np.random.binomial(shots, p0).astype(np.float64)
    gzz = 2.0 * (counts_0 / shots) - 1.0
    return gzz.astype(np.float32)


def process_training_data(
    data_dir: str = "training_data",
    t_evolution: float = 30e-6,
    shots: int = 500,
) -> None:
    """Generate gzz_grid_*.npy for all density_grid_*.npy in data_dir."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted(
        f for f in os.listdir(data_dir)
        if f.startswith("density_grid_") and f.endswith(".npy")
    )
    print(f"Generating Gzz for {len(files)} samples (vectorized, no Qiskit)...")

    for i, fname in enumerate(files):
        sid = fname.replace("density_grid_", "").replace(".npy", "")
        gzz_path = os.path.join(data_dir, f"gzz_grid_{sid}.npy")
        if os.path.exists(gzz_path):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(files)} done")
            continue

        density = np.load(os.path.join(data_dir, fname))
        gzz = density_to_gzz_grid(density, t_evolution=t_evolution, shots=shots)
        np.save(gzz_path, gzz)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)} done")

    print("All Gzz grids generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Gzz grids from density grids (fast)")
    parser.add_argument("--data_dir", default="training_data")
    parser.add_argument("--t_evolution", type=float, default=30e-6)
    parser.add_argument("--shots", type=int, default=500)
    args = parser.parse_args()

    process_training_data(
        data_dir=args.data_dir,
        t_evolution=args.t_evolution,
        shots=args.shots,
    )
