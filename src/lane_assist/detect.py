"""Hough line extraction and left/right lane fits."""

from __future__ import annotations

import cv2
import numpy as np

from lane_assist.config import PipelineConfig
from lane_assist.geometry import (
    LaneEstimate,
    Line,
    average_slope_intercept,
    estimate_confidence,
    extrapolate_line,
    lane_metrics,
    lines_from_hough,
    split_left_right,
)


def hough_lines(edges: np.ndarray, config: PipelineConfig) -> list[Line]:
    height, width = edges.shape[:2]
    scale = min(width / 1280.0, height / 720.0)
    raw = cv2.HoughLinesP(
        edges,
        rho=config.hough_rho,
        theta=config.hough_theta,
        threshold=max(8, int(round(config.hough_threshold * scale))),
        minLineLength=max(8, int(round(config.hough_min_line_length * scale))),
        maxLineGap=max(8, int(round(config.hough_max_line_gap * scale))),
    )
    return lines_from_hough(raw)


def measure_fits(
    lines: list[Line],
    image_width: int,
    config: PipelineConfig,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, int, int]:
    """Return current-frame ``(left_fit, right_fit, n_left, n_right)``."""
    left_segs, right_segs = split_left_right(
        lines,
        image_width,
        config.min_abs_slope,
        config.max_abs_slope,
        config.center_deadzone_ratio,
    )
    return (
        average_slope_intercept(left_segs),
        average_slope_intercept(right_segs),
        len(left_segs),
        len(right_segs),
    )


def estimate_from_fits(
    left_fit: tuple[float, float] | None,
    right_fit: tuple[float, float] | None,
    image_shape: tuple[int, ...],
    config: PipelineConfig,
    n_left: int = 0,
    n_right: int = 0,
) -> LaneEstimate:
    """Extrapolate lane segments and compute lateral offset / confidence."""
    height, width = image_shape[:2]
    y_top = height * config.roi_top_ratio
    y_bottom = height * config.roi_bottom_ratio
    left_line = extrapolate_line(left_fit, y_top, y_bottom, width) if left_fit else None
    right_line = extrapolate_line(right_fit, y_top, y_bottom, width) if right_fit else None
    offset_px, lane_width, offset_norm = lane_metrics(left_line, right_line, width, y_bottom)
    both = left_line is not None and right_line is not None
    return LaneEstimate(
        left=left_line,
        right=right_line,
        center_offset_px=offset_px,
        lane_width_px=lane_width,
        offset_lane_widths=offset_norm,
        confidence=estimate_confidence(n_left, n_right, both),
        supporting_left=n_left,
        supporting_right=n_right,
    )


def fit_lanes(
    lines: list[Line],
    image_shape: tuple[int, ...],
    config: PipelineConfig,
) -> tuple[LaneEstimate, tuple[float, float] | None, tuple[float, float] | None]:
    """Measure this frame and build an unsmoothed ``LaneEstimate``."""
    width = image_shape[1]
    left_fit, right_fit, n_left, n_right = measure_fits(lines, width, config)
    estimate = estimate_from_fits(left_fit, right_fit, image_shape, config, n_left, n_right)
    return estimate, left_fit, right_fit
