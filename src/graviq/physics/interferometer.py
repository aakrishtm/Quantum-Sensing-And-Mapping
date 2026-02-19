"""
Simple Mach-Zehnder-type atom-interferometer forward model for 2D acceleration
or gravity-anomaly grids. Phase shift Δφ = k_eff·a·T² (leading order); optional
wrap to [-π, π] and phase-to-signal mapping give normalized readout in [0, 1].
"""

from __future__ import annotations

import math
import numpy as np


def _backend(x) -> str:
    """Detect backend: 'torch' or 'numpy'."""
    if hasattr(x, "numpy") and callable(getattr(x, "numpy")):
        return "torch"
    mod = type(x).__module__
    if mod is not None and mod.startswith("torch"):
        return "torch"
    return "numpy"


def compute_phase_shift(a, k_eff: float, T: float):
    """
    Phase shift Δφ = k_eff * a * T² (Mach-Zehnder 3-pulse AI, leading order).
    Returns same shape/dtype/backend as a.
    """
    bk = _backend(a)
    if bk == "numpy":
        return _compute_phase_shift_numpy(a, k_eff, T)
    return _compute_phase_shift_torch(a, k_eff, T)


def _compute_phase_shift_numpy(a: np.ndarray, k_eff: float, T: float) -> np.ndarray:
    dtype = np.float64 if a.dtype == np.float64 else np.float32
    phi = (k_eff * a.astype(dtype) * (T ** 2))
    return phi.astype(a.dtype)


def _compute_phase_shift_torch(a, k_eff: float, T: float):
    import torch
    dtype = a.dtype
    phi = k_eff * a.to(torch.float64) * (T ** 2)
    return phi.to(dtype)


def phase_to_signal(phi, contrast: float = 0.8, phi0: float = 0.0, mode: str = "cos"):
    """
    Map phase to normalized signal in [0, 1].
    cos: s = 0.5 * (1 + contrast * cos(phi + phi0))
    sin: s = 0.5 * (1 + contrast * sin(phi + phi0))
    """
    bk = _backend(phi)
    if bk == "numpy":
        return _phase_to_signal_numpy(phi, contrast, phi0, mode)
    return _phase_to_signal_torch(phi, contrast, phi0, mode)


def _phase_to_signal_numpy(phi: np.ndarray, contrast: float, phi0: float, mode: str) -> np.ndarray:
    p = phi.astype(np.float64) + phi0
    if mode == "sin":
        s = 0.5 * (1.0 + contrast * np.sin(p))
    else:
        s = 0.5 * (1.0 + contrast * np.cos(p))
    return np.clip(s, 0.0, 1.0).astype(phi.dtype)


def _phase_to_signal_torch(phi, contrast: float, phi0: float, mode: str):
    import torch
    p = phi.to(torch.float64) + phi0
    if mode == "sin":
        s = 0.5 * (1.0 + contrast * torch.sin(p))
    else:
        s = 0.5 * (1.0 + contrast * torch.cos(p))
    return torch.clamp(s, 0.0, 1.0).to(phi.dtype)


def wrap_phase(phi):
    """Wrap phase to [-π, π]."""
    bk = _backend(phi)
    if bk == "numpy":
        return _wrap_phase_numpy(phi)
    return _wrap_phase_torch(phi)


def _wrap_phase_numpy(phi: np.ndarray) -> np.ndarray:
    out = np.remainder(phi.astype(np.float64) + math.pi, 2 * math.pi) - math.pi
    return out.astype(phi.dtype)


def _wrap_phase_torch(phi):
    import torch
    out = torch.remainder(phi.to(torch.float64) + math.pi, 2 * math.pi) - math.pi
    return out.to(phi.dtype)


def default_interferometer_cfg():
    """
    Safe default config for apply_interferometer_model.
    k_eff and T control dynamic range; if phase saturates, reduce k_eff or T.
    """
    return {
        "k_eff": 1e7,
        "T": 0.1,
        "contrast": 0.8,
        "phi0": 0.0,
        "wrap": True,
        "signal_mode": "cos",
        "output": "signal",
    }


def apply_interferometer_model(x, cfg: dict, seed=None):
    """
    End-to-end: acceleration/delta-g grid -> phase -> optional wrap -> phase or signal.
    cfg must contain k_eff, T; optional: contrast, phi0, wrap, signal_mode, output.
    """
    k_eff = cfg.get("k_eff")
    T = cfg.get("T")
    if k_eff is None or T is None:
        raise ValueError("apply_interferometer_model requires cfg['k_eff'] and cfg['T']")

    phi = compute_phase_shift(x, k_eff, T)

    if cfg.get("wrap", True):
        phi = wrap_phase(phi)

    if cfg.get("output", "signal") == "phase":
        return phi

    contrast = cfg.get("contrast", 0.8)
    phi0 = cfg.get("phi0", 0.0)
    signal_mode = cfg.get("signal_mode", "cos")
    return phase_to_signal(phi, contrast=contrast, phi0=phi0, mode=signal_mode)
