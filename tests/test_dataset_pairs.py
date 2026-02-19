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
from graviq.physics import default_interferometer_cfg


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


class TestDatasetSensorNoise(unittest.TestCase):
    """Test that sensor noise is applied only to input, shape preserved, mask unchanged."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        shape = (60, 150)
        for i in (1, 2):
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

    def tearDown(self):
        import shutil
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_noise_changes_input_shape_unchanged(self):
        """With sensor_noise=True, output shape is unchanged."""
        ds = TunnelDataset(
            self.temp_dir,
            sensor_noise=True,
            sensor_noise_cfg={"gaussian_sigma": 0.1},
        )
        sample = ds[0]
        self.assertEqual(sample["input"].shape, (1, 60, 150))
        self.assertEqual(sample["mask"].shape, (1, 60, 150))

    def test_noise_changes_values(self):
        """With sensor_noise=True, input values differ from no-noise."""
        ds_no_noise = TunnelDataset(self.temp_dir)
        ds_noise = TunnelDataset(
            self.temp_dir,
            sensor_noise=True,
            sensor_noise_cfg={"gaussian_sigma": 0.2},
            seed=42,
        )
        clean = ds_no_noise[0]["input"]
        noisy = ds_noise[0]["input"]
        self.assertFalse(
            np.allclose(clean.numpy(), noisy.numpy()),
            "Noisy input should differ from clean",
        )

    def test_mask_unchanged_by_noise(self):
        """Mask is identical with or without sensor_noise."""
        ds_no_noise = TunnelDataset(self.temp_dir)
        ds_noise = TunnelDataset(
            self.temp_dir,
            sensor_noise=True,
            sensor_noise_cfg={"gaussian_sigma": 0.1},
            seed=99,
        )
        mask_clean = ds_no_noise[0]["mask"]
        mask_noisy = ds_noise[0]["mask"]
        np.testing.assert_array_almost_equal(
            mask_clean.numpy(), mask_noisy.numpy(),
            err_msg="Mask must be unchanged when sensor_noise is applied",
        )

    def test_noise_deterministic_with_seed(self):
        """Same seed+idx yields same noisy input."""
        ds = TunnelDataset(
            self.temp_dir,
            sensor_noise=True,
            sensor_noise_cfg={"gaussian_sigma": 0.1},
            seed=123,
        )
        a = ds[0]["input"]
        b = ds[0]["input"]
        np.testing.assert_array_almost_equal(
            a.numpy(), b.numpy(),
            err_msg="Same index and seed should give identical noisy input",
        )


class TestDatasetInterferometer(unittest.TestCase):
    """When interferometer config is enabled, input is interferometer readout in [0,1], differing from raw grid."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        shape = (60, 150)
        for i in (1, 2):
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

    def tearDown(self):
        import shutil
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_interferometer_output_diffs_from_raw_and_in_01(self):
        """With interferometer_cfg set and no sensor noise, input differs from raw grid and is in [0, 1]."""
        raw_path = os.path.join(self.temp_dir, "density_grid_001.npy")
        raw_grid = np.load(raw_path).astype(np.float32)

        ds = TunnelDataset(
            self.temp_dir,
            interferometer_cfg=default_interferometer_cfg(),
            sensor_noise=False,
        )
        sample = ds[0]
        input_grid = sample["input"].squeeze(0).numpy()  # (H, W)

        self.assertFalse(
            np.allclose(raw_grid, input_grid),
            "Interferometer readout should differ from raw density grid",
        )
        self.assertGreaterEqual(float(np.min(input_grid)), 0.0, "Interferometer signal should be >= 0")
        self.assertLessEqual(float(np.max(input_grid)), 1.0, "Interferometer signal should be <= 1")


if __name__ == "__main__":
    unittest.main()
