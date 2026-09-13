"""Hough + fit tests on in-memory synthetic frames."""

import numpy as np

from lane_assist.config import PipelineConfig
from lane_assist.detect import fit_lanes, hough_lines, measure_fits
from lane_assist.geometry import Line
from lane_assist.preprocess import edge_map
from lane_assist.synthetic import render_scene


def test_hough_finds_segments_on_synthetic_road() -> None:
    frame = render_scene(960, 540, seed=11)
    edges = edge_map(frame, PipelineConfig())
    lines = hough_lines(edges, PipelineConfig())
    assert len(lines) >= 4


def test_fit_lanes_recovers_both_sides_on_straight_road() -> None:
    frame = render_scene(1280, 720, offset_px=0.0, curvature=0.0, seed=1)
    edges = edge_map(frame, PipelineConfig())
    estimate, left_fit, right_fit = fit_lanes(hough_lines(edges, PipelineConfig()), frame.shape, PipelineConfig())
    assert left_fit is not None and right_fit is not None
    assert estimate.both_lanes
    assert estimate.center_offset_px is not None
    assert abs(estimate.offset_lane_widths or 99) < 0.12
    assert estimate.confidence > 0.5


def test_fit_lanes_on_gentle_curve() -> None:
    frame = render_scene(1280, 720, offset_px=22.0, curvature=0.10, lighting="day", seed=2)
    edges = edge_map(frame, PipelineConfig())
    estimate, _, _ = fit_lanes(hough_lines(edges, PipelineConfig()), frame.shape, PipelineConfig())
    assert estimate.both_lanes


def test_rightward_scene_has_positive_offset() -> None:
    frame = render_scene(1280, 720, offset_px=36.0, curvature=0.0, seed=5)
    edges = edge_map(frame, PipelineConfig())
    estimate, _, _ = fit_lanes(hough_lines(edges, PipelineConfig()), frame.shape, PipelineConfig())
    assert estimate.both_lanes
    assert estimate.center_offset_px is not None
    assert estimate.center_offset_px > 0


def test_measure_fits_ignores_empty() -> None:
    left, right, n_left, n_right = measure_fits([], 1280, PipelineConfig())
    assert left is None and right is None
    assert n_left == 0 and n_right == 0


def test_black_frame_has_no_lock() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    edges = edge_map(frame, PipelineConfig())
    estimate, left, right = fit_lanes(hough_lines(edges, PipelineConfig()), frame.shape, PipelineConfig())
    assert left is None and right is None
    assert not estimate.both_lanes
    assert estimate.confidence == 0.0


def test_line_list_roundtrip_from_handcrafted_segments() -> None:
    # Two clean markings that should classify without Hough.
    lines = [
        Line(220, 680, 540, 430),
        Line(1060, 680, 740, 430),
    ]
    estimate, left, right = fit_lanes(lines, (720, 1280, 3), PipelineConfig())
    assert left is not None and right is not None
    assert estimate.both_lanes
