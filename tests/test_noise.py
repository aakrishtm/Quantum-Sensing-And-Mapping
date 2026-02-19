"""Unit tests for graviq.physics.noise transforms."""

import os
import sys

import numpy as np
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from graviq.physics.noise import (
    add_gaussian_noise,
    add_shot_noise,
    add_low_freq_drift,
    apply_sensor_model,
    add_blur,
)

SHAPE = (60, 150)


class TestNoiseShape(unittest.TestCase):
    """Shape preservation for all functions."""

    def test_add_gaussian_noise_shape(self):
        x = np.random.rand(*SHAPE).astype(np.float32)
        out = add_gaussian_noise(x, 0.1)
        self.assertEqual(out.shape, x.shape)

    def test_add_shot_noise_shape(self):
        x = np.random.rand(*SHAPE).astype(np.float32)
        out = add_shot_noise(x, 100.0)
        self.assertEqual(out.shape, x.shape)

    def test_add_low_freq_drift_shape(self):
        x = np.random.rand(*SHAPE).astype(np.float32)
        out = add_low_freq_drift(x, 0.1, kernel_size=15)
        self.assertEqual(out.shape, x.shape)

    def test_add_blur_shape(self):
        x = np.random.rand(*SHAPE).astype(np.float32)
        out = add_blur(x, 1.0)
        self.assertEqual(out.shape, x.shape)

    def test_apply_sensor_model_shape(self):
        x = np.random.rand(*SHAPE).astype(np.float32)
        cfg = {"gaussian_sigma": 0.05}
        out = apply_sensor_model(x, cfg)
        self.assertEqual(out.shape, x.shape)


class TestDeterminism(unittest.TestCase):
    """Determinism with fixed seeds."""

    def test_add_gaussian_noise_deterministic_numpy(self):
        x = np.random.rand(*SHAPE).astype(np.float32)
        out1 = add_gaussian_noise(x, 0.1, seed=42)
        out2 = add_gaussian_noise(x, 0.1, seed=42)
        np.testing.assert_array_almost_equal(out1, out2)

    def test_add_gaussian_noise_deterministic_torch(self):
        import torch
        x = torch.rand(*SHAPE)
        out1 = add_gaussian_noise(x, 0.1, seed=42)
        out2 = add_gaussian_noise(x, 0.1, seed=42)
        self.assertTrue(torch.allclose(out1, out2))

    def test_apply_sensor_model_deterministic(self):
        x = np.random.rand(*SHAPE).astype(np.float32)
        cfg = {"gaussian_sigma": 0.1, "shot_scale": 50.0}
        out1 = apply_sensor_model(x, cfg, seed=123)
        out2 = apply_sensor_model(x, cfg, seed=123)
        np.testing.assert_array_almost_equal(out1, out2)


class TestSigmaIncreasesVariance(unittest.TestCase):
    """Higher sigma gives higher variance for add_gaussian_noise."""

    def test_sigma_increases_variance(self):
        x = np.random.rand(*SHAPE).astype(np.float32)
        out_lo = add_gaussian_noise(x, 0.01, seed=1)
        out_hi = add_gaussian_noise(x, 0.5, seed=1)
        var_lo = np.var(out_lo - x)
        var_hi = np.var(out_hi - x)
        self.assertGreater(var_hi, var_lo)


class TestStrengthIncreasesVariance(unittest.TestCase):
    """Higher strength gives higher variance for add_low_freq_drift."""

    def test_strength_increases_variance(self):
        x = np.zeros(SHAPE, dtype=np.float32)  # zeros so drift dominates
        np.random.seed(99)
        out_lo = add_low_freq_drift(x, 0.01, kernel_size=15)
        np.random.seed(99)
        out_hi = add_low_freq_drift(x, 0.5, kernel_size=15)
        self.assertGreater(np.var(out_hi), np.var(out_lo))


class TestTorchNumpyConsistency(unittest.TestCase):
    """Both backends produce valid outputs with expected statistics."""

    def test_gaussian_both_backends_valid(self):
        """Torch and numpy both add zero-mean noise with correct variance."""
        import torch
        x_np = np.ones(SHAPE, dtype=np.float32)
        x_torch = torch.ones(*SHAPE)
        out_np = add_gaussian_noise(x_np, 0.1, seed=42)
        out_torch = add_gaussian_noise(x_torch, 0.1, seed=42)
        # RNGs differ across backends; just check noise stats
        noise_np = out_np - x_np
        noise_torch = (out_torch - x_torch).cpu().numpy()
        self.assertAlmostEqual(np.mean(noise_np), 0.0, places=1)
        self.assertAlmostEqual(np.mean(noise_torch), 0.0, places=1)
        self.assertAlmostEqual(np.std(noise_np), 0.1, places=1)
        self.assertAlmostEqual(np.std(noise_torch), 0.1, places=1)


class TestEmptyConfig(unittest.TestCase):
    """Empty config returns x unchanged (identity)."""

    def test_empty_config_returns_unchanged(self):
        x = np.random.rand(*SHAPE).astype(np.float32)
        out = apply_sensor_model(x, {})
        np.testing.assert_array_almost_equal(out, x)


if __name__ == "__main__":
    unittest.main()
