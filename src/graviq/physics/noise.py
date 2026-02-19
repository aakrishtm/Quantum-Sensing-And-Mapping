"""
Numpy/torch-agnostic noise transforms for 2D grids.
Dispatches to torch or numpy based on input type.
"""

from __future__ import annotations

import numpy as np

# Lazy torch import
def _import_torch():
    import torch
    return torch


def _backend(x) -> str:
    """Detect backend: 'torch' or 'numpy'."""
    if hasattr(x, "numpy") and callable(getattr(x, "numpy")):
        return "torch"
    mod = type(x).__module__
    if mod is not None and mod.startswith("torch"):
        return "torch"
    return "numpy"


def _gaussian_kernel_1d(sigma: float, length: int) -> np.ndarray:
    """Build 1D Gaussian kernel (odd length)."""
    if length % 2 == 0:
        length += 1
    center = (length - 1) / 2.0
    i = np.arange(length, dtype=np.float64)
    k = np.exp(-((i - center) ** 2) / (2 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


def add_blur(x, sigma: float):
    """
    Gaussian blur via separable 1D convolution.
    x: 2D array (H, W) - numpy or torch.
    sigma: blur radius.
    """
    if sigma <= 0:
        return x.copy() if _backend(x) == "numpy" else x.clone()
    bk = _backend(x)
    if bk == "numpy":
        return _add_blur_numpy(x, sigma)
    return _add_blur_torch(x, sigma)


def _add_blur_numpy(x: np.ndarray, sigma: float) -> np.ndarray:
    H, W = x.shape
    klen = min(int(6 * sigma) + 1, min(H, W))
    if klen % 2 == 0:
        klen += 1
    sigma_eff = max((klen - 1) / 6.0, 1e-8)
    kernel = _gaussian_kernel_1d(sigma_eff, klen)
    out = np.apply_along_axis(
        lambda a: np.convolve(a, kernel, mode="same"),
        axis=1,
        arr=x.astype(np.float64),
    )
    out = np.apply_along_axis(
        lambda a: np.convolve(a, kernel, mode="same"),
        axis=0,
        arr=out,
    )
    return out.astype(x.dtype)


def _add_blur_torch(x, sigma: float):
    import torch
    import torch.nn.functional as F

    H, W = x.shape
    klen = min(int(6 * sigma) + 1, min(H, W))
    if klen % 2 == 0:
        klen += 1
    sigma_eff = max((klen - 1) / 6.0, 1e-8)
    kernel_np = _gaussian_kernel_1d(sigma_eff, klen)
    k1 = torch.from_numpy(kernel_np).to(x.device).to(x.dtype)
    k2d = torch.outer(k1, k1).unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
    # x: (H,W) -> (1,1,H,W)
    y = x.unsqueeze(0).unsqueeze(0)
    y = F.conv2d(y, k2d, padding=klen // 2)
    return y.squeeze(0).squeeze(0)


def add_gaussian_noise(x, sigma: float, seed=None):
    """Add N(0, sigma²) noise. Returns a copy with noise added."""
    bk = _backend(x)
    if bk == "numpy":
        return _add_gaussian_noise_numpy(x, sigma, seed)
    return _add_gaussian_noise_torch(x, sigma, seed)


def _add_gaussian_noise_numpy(x: np.ndarray, sigma: float, seed=None) -> np.ndarray:
    if seed is not None:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, sigma, x.shape).astype(x.dtype)
    else:
        noise = np.random.normal(0, sigma, x.shape).astype(x.dtype)
    return (x.astype(np.float64) + noise).astype(x.dtype)


def _add_gaussian_noise_torch(x, sigma: float, seed=None):
    import torch
    if seed is not None:
        gen = torch.Generator(device=x.device).manual_seed(seed)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen) * sigma
    else:
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * sigma
    return (x + noise).clone()


def add_shot_noise(x, scale: float, seed=None):
    """
    Normalize x to [0,1], sample Poisson(scale * normalized), rescale back.
    """
    bk = _backend(x)
    if bk == "numpy":
        return _add_shot_noise_numpy(x, scale, seed)
    return _add_shot_noise_torch(x, scale, seed)


def _add_shot_noise_numpy(x: np.ndarray, scale: float, seed=None) -> np.ndarray:
    x = x.astype(np.float64)
    x_clip = np.clip(x, 0, None)
    m = float(np.max(x_clip)) if np.max(x_clip) > 0 else 1.0
    norm = x_clip / m
    lam = scale * norm
    if seed is not None:
        rng = np.random.default_rng(seed)
        counts = rng.poisson(lam)
    else:
        counts = np.random.poisson(lam)
    out = counts.astype(np.float64) / scale
    return (out * m).astype(x.dtype)


def _add_shot_noise_torch(x, scale: float, seed=None):
    import torch
    x = x.to(torch.float64)
    x_clip = torch.clamp(x, min=0)
    m = float(x_clip.max()) if x_clip.max() > 0 else 1.0
    norm = x_clip / m
    lam = scale * norm
    if seed is not None:
        gen = torch.Generator(device=x.device).manual_seed(seed)
        counts = torch.poisson(lam.clamp(min=1e-10), generator=gen)
    else:
        counts = torch.poisson(lam.clamp(min=1e-10))
    out = counts.to(torch.float64) / scale
    return (out * m).to(x.dtype)


def add_low_freq_drift(x, strength: float, kernel_size: int = 15):
    """
    Generate smooth random field via Gaussian blur of white noise,
    scale by strength, add to x. Uses RNG state set by caller (e.g. apply_sensor_model).
    """
    bk = _backend(x)
    if bk == "numpy":
        return _add_low_freq_drift_numpy(x, strength, kernel_size)
    return _add_low_freq_drift_torch(x, strength, kernel_size)


def _add_low_freq_drift_numpy(x: np.ndarray, strength: float, kernel_size: int) -> np.ndarray:
    if strength <= 0:
        return x.copy()
    noise = np.random.randn(*x.shape).astype(np.float64)
    sigma = max((kernel_size - 1) / 6.0, 1e-8)
    klen = min(kernel_size, min(x.shape[0], x.shape[1]))
    if klen % 2 == 0:
        klen += 1
    kernel = _gaussian_kernel_1d(sigma, klen)
    blurred = np.apply_along_axis(
        lambda a: np.convolve(a, kernel, mode="same"),
        axis=1,
        arr=noise,
    )
    blurred = np.apply_along_axis(
        lambda a: np.convolve(a, kernel, mode="same"),
        axis=0,
        arr=blurred,
    )
    return (x.astype(np.float64) + strength * blurred).astype(x.dtype)


def _add_low_freq_drift_torch(x, strength: float, kernel_size: int):
    import torch

    if strength <= 0:
        return x.clone()
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype)
    sigma = max((kernel_size - 1) / 6.0, 1e-8)
    klen = min(kernel_size, min(x.shape[0], x.shape[1]))
    if klen % 2 == 0:
        klen += 1
    kernel_np = _gaussian_kernel_1d(sigma, klen)
    k1 = torch.from_numpy(kernel_np).to(x.device).to(x.dtype)
    k2d = torch.outer(k1, k1).unsqueeze(0).unsqueeze(0)
    y = noise.unsqueeze(0).unsqueeze(0)
    blurred = torch.nn.functional.conv2d(y, k2d, padding=klen // 2)
    blurred = blurred.squeeze(0).squeeze(0)
    return (x + strength * blurred).clone()


def apply_sensor_model(x, cfg: dict, seed=None):
    """
    Compose: x -> blur -> drift -> gaussian -> shot.
    cfg keys: blur_sigma, drift_strength, drift_kernel_size, gaussian_sigma, shot_scale.
    Omit or 0 to skip a stage.
    """
    if seed is not None:
        bk = _backend(x)
        if bk == "numpy":
            np.random.seed(seed)
        else:
            import torch
            torch.manual_seed(seed)

    out = x
    blur_sigma = cfg.get("blur_sigma", 0)
    if blur_sigma > 0:
        out = add_blur(out, blur_sigma)

    drift_strength = cfg.get("drift_strength", 0)
    drift_kernel_size = int(cfg.get("drift_kernel_size", 15))
    if drift_strength > 0:
        out = add_low_freq_drift(out, drift_strength, drift_kernel_size)

    gaussian_sigma = cfg.get("gaussian_sigma", 0)
    if gaussian_sigma > 0:
        out = add_gaussian_noise(out, gaussian_sigma, seed)

    shot_scale = cfg.get("shot_scale", 0)
    if shot_scale > 0:
        out = add_shot_noise(out, shot_scale, seed)

    return out
