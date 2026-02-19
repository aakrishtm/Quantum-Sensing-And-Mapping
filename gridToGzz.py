import numpy as np
from qiskit import QuantumCircuit, transpile
try:
    # Try newer Qiskit Aer API
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, phase_damping_error
except ImportError:
    # Fall back to older API
    from qiskit.providers.aer import AerSimulator
    from qiskit.providers.aer.noise import NoiseModel
    from qiskit.providers.aer.noise.errors import phase_damping_error

# ---------------------------------------------------------------------
# 1. Load your density grid
# ---------------------------------------------------------------------

# Example placeholder; replace with your real grid:
density_grid = np.random.rand(100, 100)  # shape (N, M)

# ---------------------------------------------------------------------
# 2. Ramsey pulse sequence (gives Gzz naturally)
# ---------------------------------------------------------------------
def ramsey_circuit(t):
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.delay(int(t * 1e9), 0, unit="ns")
    qc.h(0)
    qc.measure(0, 0)
    return qc


# ---------------------------------------------------------------------
# 3. Convert density -> phase damping probability
#    p = 1 - exp(-rho * t)
# ---------------------------------------------------------------------
def damping_probability(rho, t):
    return 1 - np.exp(-rho * t)


# ---------------------------------------------------------------------
# 4. Compute Gzz for a single grid point using Qiskit simulation
# ---------------------------------------------------------------------
def compute_Gzz_at(rho_value, t):
    # Build noise channel
    p = damping_probability(rho_value, t)
    dephasing = phase_damping_error(p)

    # Build simulator with this noise
    backend = AerSimulator()
    
    # Create noise model and add dephasing error
    noise_model = NoiseModel()
    # Add error to qubit 0 for id and delay gates
    noise_model.add_quantum_error(dephasing, ["id", "delay"], [0])

    # Build circuit
    qc = ramsey_circuit(t)
    qc = transpile(qc, backend)

    # Run
    result = backend.run(qc, noise_model=noise_model, shots=5000).result()
    counts = result.get_counts()

    # Compute expectation value <Z> = P(0) - P(1)
    P0 = counts.get('0', 0) / 5000
    P1 = counts.get('1', 0) / 5000
    Gzz = P0 - P1
    return Gzz


# ---------------------------------------------------------------------
# 5. Loop over entire density grid
# ---------------------------------------------------------------------
t_evolution = 30e-6   # seconds (tune this!)
N, M = density_grid.shape
Gzz_grid = np.zeros((N, M))

for i in range(N):
    for j in range(M):
        rho = density_grid[i, j]
        Gzz_grid[i, j] = compute_Gzz_at(rho, t_evolution)

print("Generated Gzz grid with shape:", Gzz_grid.shape)
