"""Classical baseline detectors (no ML)."""

from graviq.baselines.threshold import (
    threshold_detector,
    morphology_postprocess,
    baseline_predict,
)

__all__ = [
    "threshold_detector",
    "morphology_postprocess",
    "baseline_predict",
]
