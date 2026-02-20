# import numpy as np
# import os

# def density_to_gzz_grid(density_grid, t_evolution=30e-6, shots=500):
#     """
#     Vectorized quantum simulation. Runs in 0.001 seconds.
#     """
#     # Analytical Ramsey phase damping math
#     P0_exact = (1.0 + np.exp(-density_grid * t_evolution)) / 2.0
    
#     # Simulate quantum shot noise using binomial distribution
#     counts_0 = np.random.binomial(shots, P0_exact)
    
#     # Calculate Gzz
#     P0_measured = counts_0 / shots
#     Gzz_grid = 2.0 * P0_measured - 1.0 # Equivalent to P0 - P1
    
#     return Gzz_grid.astype(np.float32)

# def process_training_data(data_dir='training_data', t_evolution=30e-6, shots=500):
#     density_files = sorted([f for f in os.listdir(data_dir) if f.startswith('density_grid_') and f.endswith('.npy')])
    
#     print(f"Found {len(density_files)} density grids. Starting vectorized processing...")
    
#     for i, filename in enumerate(density_files):
#         sample_id = filename.replace('density_grid_', '').replace('.npy', '')
#         density_path = os.path.join(data_dir, filename)
#         gzz_path = os.path.join(data_dir, f'gzz_grid_{sample_id}.npy')
        
#         if os.path.exists(gzz_path):
#             continue
#         # Load, Process, and Save - This will now be near-instant
#         density_grid = np.load(density_path)
#         gzz_grid = density_to_gzz_grid(density_grid, t_evolution, shots)
#         np.save(gzz_path, gzz_grid)
        
#         if (i + 1) % 10 == 0 or i == 0:
#             print(f"[{i+1}/{len(density_files)}] Processed sample {sample_id}")

#     print("All Gzz grids generated successfully!")

# if __name__ == "__main__":
#     process_training_data()

import numpy as np
import os
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, phase_damping_error

def get_batch_gzz(density_values, t_evolution=30e-6, shots=500):
    # CRITICAL: Re-enable GPU device here
    backend = AerSimulator(method='density_matrix', device='GPU')
    
    rows, cols = density_values.shape
    flat_density = density_values.flatten()
    results_gzz = []

    # Process in chunks of 1000 pixels to keep the GPU pipeline saturated but safe
    chunk_size = 1000
    for i in range(0, len(flat_density), chunk_size):
        chunk_rho = flat_density[i:i+chunk_size]
        circuits = []
        
        for rho in chunk_rho:
            qc = QuantumCircuit(1, 1)
            qc.h(0)
            qc.delay(int(t_evolution * 1e9), 0, unit="ns")
            qc.h(0)
            qc.measure(0, 0)
            
            p = 1 - np.exp(-rho * t_evolution)
            error = phase_damping_error(np.clip(p, 0, 1))
            noise_model = NoiseModel()
            noise_model.add_quantum_error(error, ["delay"], [0])
            
            # Transpile for the GPU backend
            qc_t = transpile(qc, backend)
            
            # Execute immediately in the chunk
            res = backend.run(qc_t, noise_model=noise_model, shots=shots).result()
            counts = res.get_counts()
            p0 = counts.get('0', 0) / shots
            results_gzz.append(2.0 * p0 - 1.0)
            
    return np.array(results_gzz).reshape(rows, cols)

def process_training_data(data_dir='training_data'):
    files = sorted([f for f in os.listdir(data_dir) if f.startswith('density_grid_')])
    print(f"Executing H100 GPU Accelerated Quantum Sim for {len(files)} samples...")
    
    for fname in files:
        sid = fname.replace('density_grid_', '').replace('.npy', '')
        gzz_path = os.path.join(data_dir, f'gzz_grid_{sid}.npy')
        
        if os.path.exists(gzz_path):
            continue

        density = np.load(os.path.join(data_dir, fname))
        gzz = get_batch_gzz(density) 
        np.save(gzz_path, gzz)
        print(f"Sample {sid} complete via H100 GPU.")

if __name__ == "__main__":
    process_training_data()