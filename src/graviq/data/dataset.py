import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, Tuple, Callable

from graviq.physics import apply_sensor_model, apply_interferometer_model

logger = logging.getLogger(__name__)


class TunnelDataset(Dataset):
    """
    Dataset for tunnel detection/segmentation from quantum sensor (Gzz) grids.

    Only samples with BOTH gzz_grid_*.npy and tunnel_mask_*.npy are included.
    Missing masks (or inputs) are skipped without crashing.

    Returns:
        - input: (1, H, W) tensor - Gzz quantum readout (or density if using legacy data)
        - mask: (1, H, W) tensor - ground truth segmentation
        - metadata: dict with labels
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        *,
        interferometer_cfg: Optional[Dict[str, Any]] = None,
        sensor_noise: bool = False,
        sensor_noise_cfg: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            data_dir: Directory containing gzz_grid_*.npy, tunnel_mask_*.npy, metadata_*.json
            transform: Optional transforms to apply
            interferometer_cfg: If set, apply atom-interferometer model (phase -> signal) before noise.
            sensor_noise: If True, apply sensor noise model to the input grid only.
            sensor_noise_cfg: Config dict for apply_sensor_model (blur_sigma, gaussian_sigma, etc.).
            seed: Optional RNG seed; noise uses seed+idx per sample when set.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.interferometer_cfg = interferometer_cfg
        self.sensor_noise = sensor_noise
        self.sensor_noise_cfg = sensor_noise_cfg if sensor_noise_cfg is not None else {}
        self.seed = seed

        # Scan data_dir: valid samples must have BOTH gzz_grid (input) and tunnel_mask
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Data directory does not exist: {data_dir}. Run: make data"
            )

        all_input_ids = set()
        valid_ids = []

        for filename in os.listdir(data_dir):
            if filename.startswith('gzz_grid_') and filename.endswith('.npy'):
                sample_id = filename.replace('gzz_grid_', '').replace('.npy', '')
                all_input_ids.add(sample_id)
                gzz_path = os.path.join(data_dir, f'gzz_grid_{sample_id}.npy')
                mask_path = os.path.join(data_dir, f'tunnel_mask_{sample_id}.npy')
                if os.path.exists(gzz_path) and os.path.exists(mask_path):
                    valid_ids.append(sample_id)

        self.sample_ids = sorted(valid_ids)
        skipped = len(all_input_ids) - len(self.sample_ids)

        if skipped > 0:
            logger.warning(
                "Found %d valid samples in %s (%d skipped: missing input or mask).",
                len(self.sample_ids), data_dir, skipped
            )
        else:
            logger.info("Found %d valid samples in %s.", len(self.sample_ids), data_dir)

        if not self.sample_ids:
            raise FileNotFoundError(
                f"No complete samples found in {data_dir}. Each sample needs "
                "gzz_grid_*.npy and tunnel_mask_*.npy. Run: make data"
            )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id = self.sample_ids[idx]

        # 1. Load the clean Gzz grid
        gzz_path = os.path.join(self.data_dir, f'gzz_grid_{sample_id}.npy')
        gzz_grid = np.load(gzz_path).astype(np.float32)

 # 2. INJECT NOISE DURING TRAINING
        # We force the model to see noise so it learns to be a "denoiser"
        # Using a random sigma makes the AI robust to different noise levels        if self.interferometer_cfg:
            gzz_grid = apply_interferometer_model(
            gzz_grid, self.interferometer_cfg, seed=None)
            gzz_grid = np.asarray(gzz_grid, dtype=np.float32)

        # Optionally apply sensor noise to input only (never to mask)
        if self.sensor_noise:
            # Deterministic per (seed, idx) pair so CI can verify reproducibility
            noise_seed = (self.seed + idx) if self.seed is not None else None
            gzz_grid = apply_sensor_model(
            gzz_grid, self.sensor_noise_cfg, seed=noise_seed)
            gzz_grid = np.asarray(gzz_grid, dtype=np.float32)

        # Load ground truth mask
        mask_path = os.path.join(self.data_dir, f'tunnel_mask_{sample_id}.npy')
        tunnel_mask = np.load(mask_path).astype(np.float32)

        # Add channel dimension: (H, W) -> (1, H, W)
        gzz_grid = gzz_grid[np.newaxis, ...]
        tunnel_mask = tunnel_mask[np.newaxis, ...]

        # Convert to tensors
        input_tensor = torch.from_numpy(gzz_grid).float()
        mask_tensor = torch.from_numpy(tunnel_mask).float()

        # Load metadata
        metadata_path = os.path.join(self.data_dir, f'metadata_{sample_id}.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            has_tunnel = bool(np.any(tunnel_mask > 0.5))
            metadata = {'has_tunnel': has_tunnel, 'num_tunnels': 1 if has_tunnel else 0}

        return {
            'input': input_tensor,
            'mask': mask_tensor,
            'has_tunnel': metadata['has_tunnel'],
            'num_tunnels': metadata['num_tunnels'],
            'sample_id': sample_id
        }


def get_dataloaders(
    data_dir: str, 
    batch_size: int = 8, 
    train_split: float = 0.8, 
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Directory with training data
        batch_size: Batch size
        train_split: Fraction of data for training (rest for validation)
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader
    """
    # Load full dataset
    full_dataset = TunnelDataset(data_dir)

    # Split into train/val
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    return train_loader, val_loader