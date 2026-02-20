import numpy as np
import os
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, phase_damping_error

def get_batch_gzz(density_values, t_evolution=30e-6, shots=500):
    # Initialize the base simulator
    backend = AerSimulator(method='density_matrix')
    
    # Dynamically select GPU if available, else fallback to CPU to save CI/CD
    if 'GPU' in backend.available_devices():
        backend.set_options(device='GPU')
    else:
        backend.set_options(device='CPU')
    
    rows, cols = density_values.shape
    flat_density = density_values.flatten()
    results_gzz = []

    # Single template circuit prevents the transpile() memory leak
    template_qc = QuantumCircuit(1, 1)
    template_qc.h(0)
    template_qc.delay(int(t_evolution * 1e9), 0, unit="ns")
    template_qc.h(0)
    template_qc.measure(0, 0)

    # Process in chunks to prevent Python memory overhead
    chunk_size = 500
    for i in range(0, len(flat_density), chunk_size):
        chunk_rho = flat_density[i:i+chunk_size]
        for rho in chunk_rho:
            p = 1 - np.exp(-rho * t_evolution)
            error = phase_damping_error(np.clip(p, 0, 1))
            noise_model = NoiseModel()
            noise_model.add_quantum_error(error, ["delay"], [0])
            
            job = backend.run(template_qc, noise_model=noise_model, shots=shots)
            counts = job.result().get_counts()
            p0 = counts.get('0', 0) / shots
            results_gzz.append(2.0 * p0 - 1.0)
            
    return np.array(results_gzz).reshape(rows, cols)

def process_training_data(data_dir='training_data'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    files = sorted([f for f in os.listdir(data_dir) if f.startswith('density_grid_')])
    print(f"H100/T4 GPU Execution: Processing {len(files)} samples...")
    
    for fname in files:
        sid = fname.replace('density_grid_', '').replace('.npy', '')
        gzz_path = os.path.join(data_dir, f'gzz_grid_{sid}.npy')
        if os.path.exists(gzz_path): continue

        density = np.load(os.path.join(data_dir, fname))
        gzz = get_batch_gzz(density) 
        np.save(gzz_path, gzz)
        print(f"Sample {sid} complete via GPU.")

if __name__ == "__main__":
    process_training_data()