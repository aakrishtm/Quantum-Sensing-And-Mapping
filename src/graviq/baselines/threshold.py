"""
Classical baseline detector: threshold + morphology.
Detects low-density anomalies (tunnels) via z-score thresholding.
"""

import numpy as np


def threshold_detector(grid: np.ndarray, z: float = 2.0) -> np.ndarray:
    """
    Anomaly score = (mean - grid) / std; mask = score > z.
    Detects low-density regions (tunnels) as anomalies.

    Args:
        grid: 2D density grid (H, W)
        z: Threshold in standard deviations

    Returns:
        Binary mask (H, W), 1 where anomaly score > z
    """
    mean = float(np.mean(grid))
    std = float(np.std(grid))
    if std < 1e-10:
        return np.zeros_like(grid, dtype=np.float32)
    score = (mean - grid) / std  # low density -> high score (tunnel)
    return (score > z).astype(np.float32)


def morphology_postprocess(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """
    Remove small connected components using BFS. Pure numpy, no scipy.

    Args:
        mask: Binary mask (H, W), values 0 or 1
        min_size: Minimum component size to keep (pixels)

    Returns:
        Binary mask with small components removed
    """
    mask = (mask > 0.5).astype(np.uint8)
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    out = np.zeros_like(mask, dtype=np.float32)

    def bfs(r: int, c: int) -> list:
        queue = [(r, c)]
        visited[r, c] = True
        comp = []
        while queue:
            r, c = queue.pop(0)
            comp.append((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        return comp

    for i in range(H):
        for j in range(W):
            if mask[i, j] and not visited[i, j]:
                comp = bfs(i, j)
                if len(comp) >= min_size:
                    for r, c in comp:
                        out[r, c] = 1.0

    return out


def baseline_predict(grid: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Combine threshold detector and morphology postprocess.

    cfg keys:
        z: Threshold (default 2.0)
        min_size: Min component size for postprocess (default 50). 0 to skip.
    """
    z = float(cfg.get("z", 2.0))
    min_size = int(cfg.get("min_size", 50))

    mask = threshold_detector(grid, z=z)
    if min_size > 0:
        mask = morphology_postprocess(mask, min_size=min_size)
    return mask
