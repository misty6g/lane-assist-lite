"""Tunable parameters for the classical lane-detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Defaults tuned for 1280x720 forward-facing synthetic road scenes.

    Values stay valid on similarly framed dashcam-style images. Narrower
    or wider cameras may need ROI and Hough tweaks.
    """

    blur_ksize: int = 5
    canny_low: int = 50
    canny_high: int = 150

    hough_rho: float = 2.0
    hough_theta: float = 3.141592653589793 / 180.0
    hough_threshold: int = 25
    hough_min_line_length: int = 45
    hough_max_line_gap: int = 80

    min_abs_slope: float = 0.35
    max_abs_slope: float = 5.0

    roi_top_ratio: float = 0.56
    roi_bottom_ratio: float = 0.97
    roi_top_width_ratio: float = 0.42
    roi_bottom_width_ratio: float = 0.94
    center_deadzone_ratio: float = 0.08

    ema_alpha: float = 0.40
    overlay_alpha: float = 0.35
    line_thickness: int = 12

    yellow_h_low: int = 15
    yellow_h_high: int = 40
    yellow_s_low: int = 70
    yellow_l_low: int = 80
    white_s_high: int = 80
    white_l_low: int = 180
