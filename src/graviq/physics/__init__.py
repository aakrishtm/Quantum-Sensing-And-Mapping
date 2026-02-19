"""Physics simulation: sensor models, noise, drift, interferometer."""

from graviq.physics.noise import (
    apply_sensor_model,
    add_gaussian_noise,
    add_shot_noise,
    add_low_freq_drift,
)
from graviq.physics.interferometer import (
    compute_phase_shift,
    phase_to_signal,
    wrap_phase,
    apply_interferometer_model,
    default_interferometer_cfg,
)

__all__ = [
    "apply_sensor_model",
    "add_gaussian_noise",
    "add_shot_noise",
    "add_low_freq_drift",
    "compute_phase_shift",
    "phase_to_signal",
    "wrap_phase",
    "apply_interferometer_model",
    "default_interferometer_cfg",
]
