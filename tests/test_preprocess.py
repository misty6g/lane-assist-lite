"""Preprocess helpers: color prior, ROI, and edge fusion."""

import numpy as np

from lane_assist.config import PipelineConfig
from lane_assist.preprocess import apply_roi, edge_map, lane_color_mask, to_grayscale


def test_grayscale_passthrough_and_convert() -> None:
    gray = np.zeros((8, 8), dtype=np.uint8)
    assert to_grayscale(gray) is gray
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr[:] = (0, 0, 255)  # red
    out = to_grayscale(bgr)
    assert out.shape == (8, 8)
    assert out.dtype == np.uint8


def test_color_mask_finds_yellow_and_white() -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[5:15, 5:15] = (0, 220, 240)  # BGR yellow-ish
    frame[20:30, 20:30] = (240, 240, 240)  # white
    mask = lane_color_mask(frame, PipelineConfig())
    assert mask[10, 10] == 255
    assert mask[25, 25] == 255
    assert mask[2, 2] == 0


def test_roi_zeros_out_sky() -> None:
    mask = np.full((200, 200), 255, dtype=np.uint8)
    cropped = apply_roi(mask, PipelineConfig())
    assert cropped[5, 100] == 0
    assert cropped[190, 100] == 255


def test_edge_map_keeps_lane_paint() -> None:
    from lane_assist.synthetic import render_scene

    frame = render_scene(640, 360, seed=7)
    edges = edge_map(frame, PipelineConfig())
    assert edges.shape == (360, 640)
    assert edges.max() == 255
    # Sky / hood should be mostly empty; roadway band should have edges.
    assert edges[:80].sum() < edges[180:300].sum()
