"""Unit tests for lane geometry helpers (no GPU, no I/O)."""

import math

import numpy as np

from lane_assist.config import PipelineConfig
from lane_assist.geometry import (
    Line,
    average_slope_intercept,
    blend_fit,
    bottom_x,
    classify_side,
    estimate_confidence,
    extrapolate_line,
    intercept,
    lane_metrics,
    line_slope,
    roi_vertices,
    slope,
    split_left_right,
    x_at_y,
)


def test_slope_horizontal_and_vertical() -> None:
    assert slope(0, 10, 20, 10) == 0.0
    assert math.isinf(slope(5, 0, 5, 20))
    assert slope(5, 20, 5, 0) == float("-inf")


def test_slope_typical_left_lane() -> None:
    # Bottom-left (200, 700) to vanishing-point-ish (560, 420): y drops as x rises.
    m = slope(200, 700, 560, 420)
    assert m < 0


def test_intercept_roundtrip() -> None:
    m = slope(100, 200, 300, 400)
    b = intercept(100, 200, m)
    assert abs((m * 300 + b) - 400) < 1e-6


def test_classify_side_left_and_right() -> None:
    assert classify_side(-0.8, midpoint_x=400, image_center_x=640) == "left"
    assert classify_side(0.8, midpoint_x=900, image_center_x=640) == "right"
    assert classify_side(-0.8, midpoint_x=900, image_center_x=640) is None
    assert classify_side(0.8, midpoint_x=200, image_center_x=640) is None
    assert classify_side(-0.8, midpoint_x=640, image_center_x=640, deadzone_px=80) is None


def test_split_rejects_near_horizontal() -> None:
    lines = [
        Line(100, 600, 200, 598),  # almost flat
        Line(200, 700, 500, 420),  # left
        Line(1100, 700, 780, 420),  # right
    ]
    left, right = split_left_right(lines, 1280, min_abs_slope=0.35, max_abs_slope=5.0)
    assert len(left) == 1
    assert len(right) == 1


def test_average_slope_intercept() -> None:
    lines = [Line(0, 10, 10, 20), Line(0, 12, 10, 22)]
    fit = average_slope_intercept(lines)
    assert fit is not None
    m, b = fit
    assert abs(m - 1.0) < 1e-6
    assert abs(b - 11.0) < 1e-6


def test_average_empty_is_none() -> None:
    assert average_slope_intercept([]) is None


def test_extrapolate_and_x_at_y() -> None:
    # y = 1.0 * x + 0  →  x = y
    line = extrapolate_line((1.0, 0.0), y_top=100, y_bottom=400, image_width=800)
    assert line is not None
    assert abs(x_at_y(line, 200) - 200) < 1e-6


def test_lane_metrics_centered() -> None:
    left = Line(200, 700, 500, 400)
    right = Line(1080, 700, 780, 400)
    offset, width, norm = lane_metrics(left, right, image_width=1280, sample_y=700)
    assert offset is not None and width is not None and norm is not None
    assert abs(offset) < 1e-6
    assert abs(width - 880) < 1e-6
    assert abs(norm) < 1e-6


def test_lane_metrics_rightward_drift_is_positive() -> None:
    # Lane center is left of the image center → camera is right of the lane.
    left = Line(100, 700, 400, 400)
    right = Line(900, 700, 600, 400)
    offset, _, _ = lane_metrics(left, right, image_width=1280, sample_y=700)
    assert offset is not None and offset > 0


def test_lane_metrics_incomplete() -> None:
    assert lane_metrics(None, Line(1, 1, 2, 2), 100, 10) == (None, None, None)


def test_confidence_bounds() -> None:
    assert estimate_confidence(0, 0, False) == 0.0
    assert 0.0 < estimate_confidence(2, 0, False) < 1.0
    assert 0.5 < estimate_confidence(4, 4, True) <= 1.0


def test_roi_vertices_are_a_trapezoid() -> None:
    poly = roi_vertices(1280, 720, PipelineConfig())
    assert poly.shape == (1, 4, 2)
    assert poly.dtype == np.int32
    xs = poly[0, :, 0]
    ys = poly[0, :, 1]
    assert ys.min() < ys.max()
    assert xs.min() < 640 < xs.max()


def test_blend_fit_ema() -> None:
    assert blend_fit(None, (1.0, 2.0), 0.5) == (1.0, 2.0)
    assert blend_fit((1.0, 2.0), None, 0.5) == (1.0, 2.0)
    blended = blend_fit((0.0, 0.0), (1.0, 10.0), 0.4)
    assert blended == (0.4, 4.0)


def test_bottom_x_picks_larger_y() -> None:
    assert bottom_x(Line(10, 700, 400, 400)) == 10
    assert bottom_x(Line(400, 400, 10, 700)) == 10


def test_line_slope_and_as_int() -> None:
    line = Line(1.2, 3.7, 5.8, 9.1)
    assert abs(line_slope(line) - slope(1.2, 3.7, 5.8, 9.1)) < 1e-12
    assert line.as_int_tuple() == (1, 4, 6, 9)
    assert line.points == ((1.2, 3.7), (5.8, 9.1))
