#!/usr/bin/env python3
"""Thin wrapper: run GraviQ inference / evaluation."""

import os
import sys
import torch
import numpy as np

from graviq.inference import load_model, predict, evaluate_dataset
from graviq.viz import visualize_prediction

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = 'checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No trained model at {checkpoint_path}")
        print("Run 'make train' first to train a model.")
        sys.exit(1)
    model = load_model(checkpoint_path, device)

    # Evaluate on dataset
    evaluate_dataset(model, 'training_data', device,
                     save_dir='evaluation_results',
                     max_samples_per_category=10)

    # Single sample prediction example
    print("\n" + "=" * 60)
    print("SINGLE SAMPLE PREDICTION")
    print("=" * 60)

    input_path = 'training_data/gzz_grid_000.npy'
    ground_truth_path = 'training_data/tunnel_mask_000.npy'
    if os.path.exists(input_path) and os.path.exists(ground_truth_path):
        input_grid = np.load(input_path)
        ground_truth = np.load(ground_truth_path)
        prob_map, binary_mask, has_tunnel = predict(model, input_grid, device)
        print(f"Detected tunnel: {has_tunnel}")
        print(f"Max probability: {prob_map.max():.4f}")
        print(f"Tunnel pixels: {binary_mask.sum()}")
        visualize_prediction(input_grid, ground_truth, prob_map, binary_mask,
                             save_path='single_prediction.png', show=False)
    else:
        print("Skipping single sample (training_data/gzz_grid_000.npy not found).")


if __name__ == "__main__":
    main()
