"""Fast analytical approximation for Gzz from density grid (no quantum simulation)."""

import numpy as np


def gzz_approximation(density_grid, t_evolution=30e-6):
    """
    Fast approximation of Gzz grid from density grid.
    Uses analytical phase-damping-style decay: Gzz \approx 2*exp(-rho*t) - 1.
    """
    # Scale so typical densities give sensible Gzz in [-1, 1]
    scale = t_evolution * 1e6  # per-second scale
    gzz_grid = 2.0 * np.exp(-density_grid * scale) - 1.0
    return gzz_grid.astype(np.float64)
