"""
Generate Gzz grids from density grids via Qiskit GPU quantum simulation.

Uses a Ramsey interferometer circuit with density-dependent phase damping to model
realistic atom-interferometer gravimetry. 
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, phase_damping_error

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
BATCH_SIZE = 256  # Circuits per batch (tune for GPU memory)
DENSITY_DECIMALS = 4  # Group pixels by rounded density to reduce unique circuits


def _ramsey_circuit(delay_ns: int) -> QuantumCircuit:
    """
    Ramsey pulse sequence: H - delay - H - measure.
    Produces Z-basis measurement with phase-dependent outcome.
    """
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.delay(delay_ns, 0, unit="ns")
    qc.h(0)
    qc.measure(0, 0)
    return qc


def _damping_probability(rho: float, t_evolution: float) -> float:
    """Phase damping probability: p = 1 - exp(-rho * t)."""
    return float(1.0 - np.exp(-float(rho) * float(t_evolution)))


def _build_simulator(device: str = "GPU") -> AerSimulator:
    """
    Instantiate AerSimulator configured for GPU.
    Requires qiskit-aer-gpu on CUDA-capable systems.
    """
    return AerSimulator(
        method="density_matrix",
        device=device,
    )


def _counts_to_gzz(counts: Dict[str, int], shots: int) -> float:
    """Compute Gzz = <Z> = (P(0) - P(1)) from measurement counts."""
    n0 = counts.get("0", 0)
    n1 = counts.get("1", 0)
    return (n0 - n1) / shots


def density_to_gzz_grid(
    density_grid: np.ndarray,
    t_evolution: float = 30e-6,
    shots: int = 500,
    batch_size: int = BATCH_SIZE,
    device: str = "GPU",
) -> np.ndarray:
    """
    Generate Gzz grid via Qiskit quantum simulation with batched execution.

    Uses Ramsey circuit + density-dependent phase damping. Circuits are grouped
    by rounded density to minimize unique (circuit, noise_model) pairs and
    maximize batch throughput on GPU.

    Returns:
        np.float32 array of shape (N_z, N_x), same as density_grid.
    """
    density_grid = np.asarray(density_grid, dtype=np.float64)
    nz, nx = density_grid.shape
    total_pixels = nz * nx

    delay_ns = int(t_evolution * 1e9)

    # Group pixel indices by quantized density to enable batching
    rho_flat = density_grid.ravel()
    rho_rounded = np.round(rho_flat, decimals=DENSITY_DECIMALS)
    unique_rhos, inverse = np.unique(rho_rounded, return_inverse=True)

    gzz_flat = np.zeros(total_pixels, dtype=np.float64)

    simulator = _build_simulator(device)
    base_circuit = _ramsey_circuit(delay_ns)
    base_circuit = transpile(base_circuit, simulator)

    for idx_rho, rho_val in enumerate(unique_rhos):
        p = _damping_probability(float(rho_val), t_evolution)
        p = np.clip(p, 0.0, 1.0)

        noise_model = NoiseModel()
        dephasing = phase_damping_error(p)
        noise_model.add_quantum_error(dephasing, ["id", "delay"], [0])

        pixel_indices = np.where(inverse == idx_rho)[0]
        n_circuits = len(pixel_indices)

        for start in range(0, n_circuits, batch_size):
            end = min(start + batch_size, n_circuits)
            batch_idx = pixel_indices[start:end]
            circuits = [base_circuit.copy() for _ in range(len(batch_idx))]

            job = simulator.run(
                circuits,
                noise_model=noise_model,
                shots=shots,
            )
            result = job.result()

            for k, idx in enumerate(batch_idx):
                counts = result.get_counts(k)
                gzz_flat[idx] = _counts_to_gzz(counts, shots)

    gzz_grid = gzz_flat.reshape(nz, nx)
    return gzz_grid.astype(np.float32)


def process_training_data(
    data_dir: str = "training_data",
    t_evolution: float = 30e-6,
    shots: int = 500,
    batch_size: int = BATCH_SIZE,
    device: str = "GPU",
) -> None:
    """Generate gzz_grid_*.npy for all density_grid_*.npy in data_dir."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted(
        f
        for f in os.listdir(data_dir)
        if f.startswith("density_grid_") and f.endswith(".npy")
    )
    print(
        f"Generating Gzz for {len(files)} samples (Qiskit Aer GPU, batch_size={batch_size})..."
    )

    for i, fname in enumerate(files):
        sid = fname.replace("density_grid_", "").replace(".npy", "")
        gzz_path = os.path.join(data_dir, f"gzz_grid_{sid}.npy")
        if os.path.exists(gzz_path):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(files)} done")
            continue

        density = np.load(os.path.join(data_dir, fname))
        gzz = density_to_gzz_grid(
            density,
            t_evolution=t_evolution,
            shots=shots,
            batch_size=batch_size,
            device=device,
        )
        np.save(gzz_path, gzz)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)} done")

    print("All Gzz grids generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Gzz grids from density grids (Qiskit GPU simulation)"
    )
    parser.add_argument("--data_dir", default="training_data")
    parser.add_argument("--t_evolution", type=float, default=30e-6)
    parser.add_argument("--shots", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--device",
        choices=["GPU", "CPU"],
        default="GPU",
        help="AerSimulator device (GPU requires qiskit-aer-gpu)",
    )
    args = parser.parse_args()

    process_training_data(
        data_dir=args.data_dir,
        t_evolution=args.t_evolution,
        shots=args.shots,
        batch_size=args.batch_size,
        device=args.device,
    )
