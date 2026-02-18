import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TunnelDataset(Dataset):
    """
    Dataset for tunnel detection/segmentation from density grids.

    Only samples with BOTH density_grid_*.npy and tunnel_mask_*.npy are included.
    Missing masks (or inputs) are skipped without crashing.

    Returns:
        - density_grid: (1, H, W) tensor - input
        - tunnel_mask: (1, H, W) tensor - ground truth segmentation
        - metadata: dict with labels
    """

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory containing density_grid_*.npy, tunnel_mask_*.npy, metadata_*.json
            transform: Optional transforms to apply
        """
        self.data_dir = data_dir
        self.transform = transform

        # Scan data_dir and collect sample_ids where BOTH input and mask exist
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Data directory does not exist: {data_dir}. Run: make data"
            )

        all_input_ids = set()
        valid_ids = []

        for filename in os.listdir(data_dir):
            if filename.startswith('density_grid_') and filename.endswith('.npy'):
                sample_id = filename.replace('density_grid_', '').replace('.npy', '')
                all_input_ids.add(sample_id)
                density_path = os.path.join(data_dir, f'density_grid_{sample_id}.npy')
                mask_path = os.path.join(data_dir, f'tunnel_mask_{sample_id}.npy')
                if os.path.exists(density_path) and os.path.exists(mask_path):
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
                "density_grid_*.npy and tunnel_mask_*.npy. Run: make data"
            )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        # Load density grid (input)
        density_path = os.path.join(self.data_dir, f'density_grid_{sample_id}.npy')
        density_grid = np.load(density_path).astype(np.float32)

        # Load tunnel mask (ground truth)
        mask_path = os.path.join(self.data_dir, f'tunnel_mask_{sample_id}.npy')
        tunnel_mask = np.load(mask_path).astype(np.float32)

        # Load metadata (optional; infer from mask if missing)
        metadata_path = os.path.join(self.data_dir, f'metadata_{sample_id}.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            has_tunnel = bool(np.any(tunnel_mask > 0.5))
            metadata = {'has_tunnel': has_tunnel, 'num_tunnels': 1 if has_tunnel else 0}

        # Add channel dimension: (H, W) -> (1, H, W)
        density_grid = density_grid[np.newaxis, ...]
        tunnel_mask = tunnel_mask[np.newaxis, ...]

        # Convert to tensors
        density_grid = torch.from_numpy(density_grid)
        tunnel_mask = torch.from_numpy(tunnel_mask)

        if self.transform:
            density_grid, tunnel_mask = self.transform(density_grid, tunnel_mask)

        return {
            'input': density_grid,
            'mask': tunnel_mask,
            'has_tunnel': metadata['has_tunnel'],
            'num_tunnels': metadata['num_tunnels'],
            'sample_id': sample_id
        }


def get_dataloaders(data_dir, batch_size=8, train_split=0.8, num_workers=0):
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
