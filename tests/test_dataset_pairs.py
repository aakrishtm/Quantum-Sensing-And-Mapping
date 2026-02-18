"""Unit tests for TunnelDataset handling of missing mask files."""

import json
import os
import tempfile
import unittest

import numpy as np

# Import after ensuring project root is on path
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from graviq.data.dataset import TunnelDataset


class TestDatasetPairs(unittest.TestCase):
    """Test that TunnelDataset skips samples with missing masks."""

    def setUp(self):
        """Create temp dir with 3 complete pairs and 2 missing masks."""
        self.temp_dir = tempfile.mkdtemp()
        shape = (60, 150)

        # 3 complete pairs: density_grid + tunnel_mask + metadata
        for i in (1, 2, 3):
            sample_id = f"{i:03d}"
            np.save(
                os.path.join(self.temp_dir, f"density_grid_{sample_id}.npy"),
                np.random.rand(*shape).astype(np.float32),
            )
            np.save(
                os.path.join(self.temp_dir, f"tunnel_mask_{sample_id}.npy"),
                (np.random.rand(*shape) > 0.9).astype(np.float32),
            )
            with open(
                os.path.join(self.temp_dir, f"metadata_{sample_id}.json"), "w"
            ) as f:
                json.dump({"has_tunnel": True, "num_tunnels": 1}, f)

        # 2 density grids with NO mask (missing tunnel_mask)
        for i in (4, 5):
            sample_id = f"{i:03d}"
            np.save(
                os.path.join(self.temp_dir, f"density_grid_{sample_id}.npy"),
                np.random.rand(*shape).astype(np.float32),
            )
            # intentionally no tunnel_mask_{sample_id}.npy

    def tearDown(self):
        import shutil
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_dataset_length_is_three(self):
        """Dataset should include only the 3 complete pairs."""
        ds = TunnelDataset(self.temp_dir)
        self.assertEqual(len(ds), 3, "Expected 3 valid samples (2 skipped for missing mask)")

    def test_getitem_works(self):
        """__getitem__ should load valid samples without crashing."""
        ds = TunnelDataset(self.temp_dir)
        for idx in range(len(ds)):
            sample = ds[idx]
            self.assertIn("input", sample)
            self.assertIn("mask", sample)
            self.assertIn("sample_id", sample)
            self.assertIn("has_tunnel", sample)
            self.assertIn("num_tunnels", sample)
            self.assertEqual(sample["input"].shape[0], 1)
            self.assertEqual(sample["mask"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
