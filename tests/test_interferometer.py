"""Unit tests for graviq.physics.interferometer."""

import math
import os
import sys

import numpy as np
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from graviq.physics.interferometer import (
    compute_phase_shift,
    phase_to_signal,
    wrap_phase,
    apply_interferometer_model,
    default_interferometer_cfg,
)

SHAPE = (60, 150)


class TestInterferometerShape(unittest.TestCase):
    """Shape preservation for all functions."""

    def test_compute_phase_shift_shape_numpy(self):
        a = np.random.rand(*SHAPE).astype(np.float32)
        out = compute_phase_shift(a, 1e7, 0.1)
        self.assertEqual(out.shape, a.shape)

    def test_compute_phase_shift_shape_torch(self):
        import torch
        a = torch.rand(*SHAPE)
        out = compute_phase_shift(a, 1e7, 0.1)
        self.assertEqual(out.shape, a.shape)

    def test_phase_to_signal_shape_numpy(self):
        phi = np.random.rand(*SHAPE).astype(np.float32)
        out = phase_to_signal(phi, contrast=0.8)
        self.assertEqual(out.shape, phi.shape)

    def test_phase_to_signal_shape_torch(self):
        import torch
        phi = torch.rand(*SHAPE)
        out = phase_to_signal(phi, contrast=0.8)
        self.assertEqual(out.shape, phi.shape)

    def test_wrap_phase_shape_numpy(self):
        phi = np.random.rand(*SHAPE).astype(np.float32)
        out = wrap_phase(phi)
        self.assertEqual(out.shape, phi.shape)

    def test_wrap_phase_shape_torch(self):
        import torch
        phi = torch.rand(*SHAPE)
        out = wrap_phase(phi)
        self.assertEqual(out.shape, phi.shape)

    def test_apply_interferometer_model_shape_numpy(self):
        x = np.random.rand(*SHAPE).astype(np.float32)
        cfg = {"k_eff": 1e7, "T": 0.1, "output": "signal"}
        out = apply_interferometer_model(x, cfg)
        self.assertEqual(out.shape, x.shape)

    def test_apply_interferometer_model_shape_torch(self):
        import torch
        x = torch.rand(*SHAPE)
        cfg = {"k_eff": 1e7, "T": 0.1, "output": "signal"}
        out = apply_interferometer_model(x, cfg)
        self.assertEqual(out.shape, x.shape)


class TestDeterministicMath(unittest.TestCase):
    """Constant input -> constant phase = k_eff * a * T**2."""

    def test_compute_phase_shift_constant_numpy(self):
        k_eff, T = 1e7, 0.1
        a = np.full(SHAPE, 1.0, dtype=np.float32)
        out = compute_phase_shift(a, k_eff, T)
        expected = k_eff * 1.0 * (T ** 2)
        self.assertTrue(np.allclose(out, expected))
        self.assertEqual(out.shape, a.shape)

    def test_compute_phase_shift_constant_torch(self):
        import torch
        k_eff, T = 1e7, 0.1
        a = torch.full(SHAPE, 1.0, dtype=torch.float32)
        out = compute_phase_shift(a, k_eff, T)
        expected = k_eff * 1.0 * (T ** 2)
        self.assertTrue(torch.allclose(out, torch.full_like(out, expected)))


class TestWrapPhase(unittest.TestCase):
    """wrap_phase outputs in [-pi, pi]."""

    def test_wrap_phase_in_range_numpy(self):
        phi = np.array([0.0, 3 * math.pi, -5 * math.pi, math.pi, -math.pi], dtype=np.float64)
        out = wrap_phase(phi)
        self.assertTrue(np.all(out >= -math.pi))
        self.assertTrue(np.all(out <= math.pi))

    def test_wrap_phase_in_range_torch(self):
        import torch
        phi = torch.tensor([0.0, 3 * math.pi, -5 * math.pi, math.pi, -math.pi], dtype=torch.float64)
        out = wrap_phase(phi)
        self.assertTrue(torch.all(out >= -math.pi).item())
        self.assertTrue(torch.all(out <= math.pi).item())


class TestSignalBounds(unittest.TestCase):
    """phase_to_signal and apply_interferometer_model(output='signal') in [0, 1]."""

    def test_phase_to_signal_bounds_numpy(self):
        phi = np.random.randn(*SHAPE).astype(np.float32) * 10
        out = phase_to_signal(phi, contrast=0.8)
        self.assertGreaterEqual(float(np.min(out)), 0.0)
        self.assertLessEqual(float(np.max(out)), 1.0)

    def test_phase_to_signal_bounds_torch(self):
        import torch
        phi = torch.randn(*SHAPE) * 10
        out = phase_to_signal(phi, contrast=0.8)
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)

    def test_apply_interferometer_model_signal_bounds_numpy(self):
        x = np.random.rand(*SHAPE).astype(np.float32)
        cfg = {"k_eff": 1e7, "T": 0.1, "output": "signal"}
        out = apply_interferometer_model(x, cfg)
        self.assertGreaterEqual(float(np.min(out)), 0.0)
        self.assertLessEqual(float(np.max(out)), 1.0)


class TestTorchNumpyConsistency(unittest.TestCase):
    """Deterministic path: torch and numpy outputs numerically close."""

    def test_compute_phase_shift_consistent(self):
        import torch
        k_eff, T = 1e5, 0.1
        a_val = 0.5
        a_np = np.full(SHAPE, a_val, dtype=np.float32)
        a_torch = torch.full(SHAPE, a_val, dtype=torch.float32)
        out_np = compute_phase_shift(a_np, k_eff, T)
        out_torch = compute_phase_shift(a_torch, k_eff, T)
        np.testing.assert_allclose(out_torch.cpu().numpy(), out_np, rtol=1e-5)

    def test_phase_to_signal_consistent(self):
        import torch
        phi_np = np.full(SHAPE, 0.5, dtype=np.float32)
        phi_torch = torch.full(SHAPE, 0.5, dtype=torch.float32)
        out_np = phase_to_signal(phi_np, contrast=0.8, phi0=0.0, mode="cos")
        out_torch = phase_to_signal(phi_torch, contrast=0.8, phi0=0.0, mode="cos")
        np.testing.assert_allclose(out_torch.cpu().numpy(), out_np, rtol=1e-5)

    def test_wrap_phase_consistent(self):
        import torch
        phi_np = np.full(SHAPE, 2.5, dtype=np.float32)
        phi_torch = torch.full(SHAPE, 2.5, dtype=torch.float32)
        out_np = wrap_phase(phi_np)
        out_torch = wrap_phase(phi_torch)
        np.testing.assert_allclose(out_torch.cpu().numpy(), out_np, rtol=1e-5)


class TestDefaultInterferometerCfg(unittest.TestCase):
    """default_interferometer_cfg returns expected keys and values."""

    def test_has_expected_keys(self):
        cfg = default_interferometer_cfg()
        expected_keys = {"k_eff", "T", "contrast", "phi0", "wrap", "signal_mode", "output"}
        self.assertEqual(set(cfg.keys()), expected_keys)

    def test_default_values(self):
        cfg = default_interferometer_cfg()
        self.assertEqual(cfg["k_eff"], 1e7)
        self.assertEqual(cfg["T"], 0.1)
        self.assertEqual(cfg["contrast"], 0.8)
        self.assertEqual(cfg["phi0"], 0.0)
        self.assertTrue(cfg["wrap"])
        self.assertEqual(cfg["signal_mode"], "cos")
        self.assertEqual(cfg["output"], "signal")


if __name__ == "__main__":
    unittest.main()
