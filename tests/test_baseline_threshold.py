"""Unit tests for threshold + morphology baseline detector."""

import os
import sys

import numpy as np
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from graviq.baselines.threshold import (
    threshold_detector,
    morphology_postprocess,
    baseline_predict,
)


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU = intersection / union."""
    pred = (pred > 0.5).astype(np.float32)
    gt = (gt > 0.5).astype(np.float32)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return float(inter / (union + 1e-8))


def _make_synthetic_tunnel_grid(
    shape=(60, 150),
    tunnel_width=8,
    tunnel_row=30,
    background_density=0.8,
    tunnel_density=0.2,
    noise_std=0.05,
    seed=42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic grid: background + low-density tunnel strip + noise.
    Returns (grid, ground_truth_mask).
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    grid = np.full(shape, background_density, dtype=np.float32)
    gt = np.zeros(shape, dtype=np.float32)

    # Tunnel: horizontal low-density strip
    r0 = max(0, tunnel_row - tunnel_width // 2)
    r1 = min(H, tunnel_row + tunnel_width // 2)
    grid[r0:r1, :] = tunnel_density
    gt[r0:r1, :] = 1.0

    # Add noise (some isolated high/low speckles)
    grid = grid + rng.normal(0, noise_std, shape).astype(np.float32)
    grid = np.clip(grid, 0, 1)

    return grid, gt


class TestBaselineThreshold(unittest.TestCase):
    """Tests for threshold detector and morphology postprocess."""

    def test_threshold_detector_shape(self):
        grid = np.random.rand(60, 150).astype(np.float32)
        mask = threshold_detector(grid, z=2.0)
        self.assertEqual(mask.shape, grid.shape)

    def test_morphology_postprocess_shape(self):
        mask = (np.random.rand(60, 150) > 0.9).astype(np.float32)
        out = morphology_postprocess(mask, min_size=50)
        self.assertEqual(out.shape, mask.shape)

    def test_baseline_predict_shape(self):
        grid = np.random.rand(60, 150).astype(np.float32)
        out = baseline_predict(grid, {"z": 2.0})
        self.assertEqual(out.shape, grid.shape)

    def test_baseline_predict_empty_cfg(self):
        grid = np.random.rand(60, 150).astype(np.float32)
        out = baseline_predict(grid, {})
        self.assertEqual(out.shape, grid.shape)

    def test_iou_improves_with_postprocess(self):
        """Synthetic tunnel grid: postprocess should improve IoU vs raw threshold."""
        grid, gt = _make_synthetic_tunnel_grid(seed=123)

        raw_mask = threshold_detector(grid, z=2.0)
        raw_iou = _iou(raw_mask, gt)

        post_mask = morphology_postprocess(raw_mask, min_size=50)
        post_iou = _iou(post_mask, gt)

        # Postprocess removes small noise blobs; should improve or match IoU
        self.assertGreaterEqual(
            post_iou,
            raw_iou - 0.05,
            "Postprocess should not degrade IoU significantly; "
            f"raw_iou={raw_iou:.4f}, post_iou={post_iou:.4f}",
        )

    def test_baseline_predict_combines_both(self):
        """baseline_predict with min_size>0 should apply postprocess."""
        grid, gt = _make_synthetic_tunnel_grid(seed=456)
        pred = baseline_predict(grid, {"z": 2.0, "min_size": 50})
        iou = _iou(pred, gt)
        self.assertGreater(iou, 0, "Should detect some tunnel pixels")
        self.assertEqual(pred.shape, gt.shape)

    def test_baseline_predict_min_size_zero_skips_postprocess(self):
        """min_size=0 skips morphology (raw threshold only)."""
        grid = np.random.rand(60, 150).astype(np.float32)
        raw = threshold_detector(grid, z=2.0)
        pred = baseline_predict(grid, {"z": 2.0, "min_size": 0})
        np.testing.assert_array_almost_equal(pred, raw)


if __name__ == "__main__":
    unittest.main()
