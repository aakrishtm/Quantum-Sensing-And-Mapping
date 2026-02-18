"""Thin wrapper: run GraviQ inference (delegates to graviq.inference)."""
import os
import torch
import numpy as np
from graviq.inference import load_model, predict, evaluate_dataset
from graviq.viz import visualize_prediction

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model('checkpoints/best_model.pth', device)
    evaluate_dataset(model, 'training_data', device,
                    save_dir='evaluation_results',
                    max_samples_per_category=10)
    print("\n" + "=" * 60)
    print("SINGLE SAMPLE PREDICTION")
    print("=" * 60)
    density_path = 'training_data/density_grid_000.npy'
    ground_truth_path = 'training_data/tunnel_mask_000.npy'
    if os.path.exists(density_path) and os.path.exists(ground_truth_path):
        density_grid = np.load(density_path)
        ground_truth = np.load(ground_truth_path)
        prob_map, binary_mask, has_tunnel = predict(model, density_grid, device)
        print(f"Detected tunnel: {has_tunnel}")
        print(f"Max probability: {prob_map.max():.4f}")
        print(f"Tunnel pixels: {binary_mask.sum()}")
        visualize_prediction(density_grid, ground_truth, prob_map, binary_mask,
                             save_path='single_prediction.png', show=False)
    else:
        print("Skipping single sample (training_data/density_grid_000.npy not found).")
