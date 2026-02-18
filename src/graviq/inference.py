"""Inference and evaluation for GraviQ."""

import os
import json
import torch
import numpy as np

from graviq.models import UNet
from graviq.data import TunnelDataset
from graviq.viz import visualize_prediction


def load_model(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load trained model from checkpoint"""
    model = UNet(in_channels=1, out_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_dice' in checkpoint:
        print(f"  Val Dice: {checkpoint['val_dice']:.4f}")
    return model


def predict(model, density_grid, device='cuda' if torch.cuda.is_available() else 'cpu', threshold=0.5):
    """
    Run inference on a single density grid.

    Args:
        model: Trained UNet model
        density_grid: (H, W) numpy array
        device: torch device
        threshold: Probability threshold for binary mask

    Returns:
        prediction_prob: (H, W) probability map
        prediction_binary: (H, W) binary mask
        has_tunnel: boolean
    """
    input_tensor = torch.from_numpy(density_grid).float()
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob_map = torch.sigmoid(output)
    prob_map = prob_map.squeeze().cpu().numpy()
    binary_mask = (prob_map > threshold).astype(np.uint8)
    has_tunnel = np.any(binary_mask)
    return prob_map, binary_mask, has_tunnel


def evaluate_dataset(model, data_dir, device='cuda' if torch.cuda.is_available() else 'cpu',
                     save_dir='evaluation_results', max_samples_per_category=10):
    """Evaluate model on entire dataset and save results"""
    os.makedirs(save_dir, exist_ok=True)
    success_dir = os.path.join(save_dir, 'successful_dice_above_0.8')
    failure_dir = os.path.join(save_dir, 'unsuccessful_dice_below_0.8')
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)

    dataset = TunnelDataset(data_dir)
    total_correct = 0
    total_samples = 0
    dice_scores = []
    iou_scores = []
    num_success_saved = 0
    num_failure_saved = 0

    print(f"Evaluating on {len(dataset)} samples...")

    for i in range(len(dataset)):
        sample = dataset[i]
        density_grid = sample['input'].squeeze().numpy()
        ground_truth = sample['mask'].squeeze().numpy()
        has_tunnel_gt = sample['has_tunnel']
        sample_id = sample['sample_id']

        prob_map, binary_mask, has_tunnel_pred = predict(
            model, density_grid, device, threshold=0.5
        )
        intersection = (binary_mask * ground_truth).sum()
        union = binary_mask.sum() + ground_truth.sum() - intersection
        iou = intersection / (union + 1e-7)
        dice = (2 * intersection) / (binary_mask.sum() + ground_truth.sum() + 1e-7)
        dice_scores.append(dice)
        iou_scores.append(iou)

        if has_tunnel_pred == has_tunnel_gt:
            total_correct += 1
        total_samples += 1

        if dice >= 0.8 and num_success_saved < max_samples_per_category:
            visualize_prediction(
                density_grid, ground_truth, prob_map, binary_mask,
                save_path=os.path.join(success_dir, f'sample_{sample_id}_dice_{dice:.3f}.png'),
                show=False, verbose=False
            )
            num_success_saved += 1
        elif dice < 0.8 and num_failure_saved < max_samples_per_category:
            visualize_prediction(
                density_grid, ground_truth, prob_map, binary_mask,
                save_path=os.path.join(failure_dir, f'sample_{sample_id}_dice_{dice:.3f}.png'),
                show=False, verbose=False
            )
            num_failure_saved += 1

    num_successful = sum(1 for d in dice_scores if d >= 0.8)
    num_unsuccessful = sum(1 for d in dice_scores if d < 0.8)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Classification Accuracy (tunnel vs no-tunnel): {100*total_correct/total_samples:.2f}% ({total_correct}/{total_samples})")
    print(f"  (Dice/IoU below are the main segmentation metrics; 100% classification on this synthetic data can indicate overfitting.)")
    print(f"Mean Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"Mean IoU Score: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    print(f"\nDice Score Distribution:")
    print(f"  Successful (≥ 0.8): {num_successful}/{total_samples} ({100*num_successful/total_samples:.1f}%)")
    print(f"  Unsuccessful (< 0.8): {num_unsuccessful}/{total_samples} ({100*num_unsuccessful/total_samples:.1f}%)")
    print(f"\nSaved Visualizations:")
    print(f"  {success_dir}: {num_success_saved} samples")
    print(f"  {failure_dir}: {num_failure_saved} samples")
    print(f"{'='*60}\n")

    results = {
        'classification_accuracy': total_correct / total_samples,
        'mean_dice': float(np.mean(dice_scores)),
        'std_dice': float(np.std(dice_scores)),
        'mean_iou': float(np.mean(iou_scores)),
        'std_iou': float(np.std(iou_scores)),
        'num_successful_dice_above_0.8': num_successful,
        'num_unsuccessful_dice_below_0.8': num_unsuccessful,
        'success_rate': num_successful / total_samples,
    }
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
