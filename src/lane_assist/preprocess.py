"""Image conditioning: blur, edges, lane-color mask, and ROI crop."""

from __future__ import annotations

import cv2
import numpy as np

from lane_assist.config import PipelineConfig
from lane_assist.geometry import roi_vertices


def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def blur(gray: np.ndarray, ksize: int) -> np.ndarray:
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(gray, (k, k), 0)


def canny_edges(gray: np.ndarray, low: int, high: int) -> np.ndarray:
    return cv2.Canny(gray, low, high)


def lane_color_mask(bgr: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Highlight yellow and white paint in HLS space."""
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    yellow = (
        (h >= config.yellow_h_low)
        & (h <= config.yellow_h_high)
        & (s >= config.yellow_s_low)
        & (l >= config.yellow_l_low)
    )
    white = (l >= config.white_l_low) & (s <= config.white_s_high)
    return np.where(yellow | white, 255, 0).astype(np.uint8)


def apply_roi(mask: np.ndarray, config: PipelineConfig) -> np.ndarray:
    height, width = mask.shape[:2]
    vertices = roi_vertices(width, height, config)
    roi = np.zeros_like(mask)
    cv2.fillPoly(roi, [vertices], 255)
    return cv2.bitwise_and(mask, roi)


def edge_map(bgr: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Combine Canny edges with a yellow/white color prior, then apply ROI."""
    gray = to_grayscale(bgr)
    smoothed = blur(gray, config.blur_ksize)
    edges = canny_edges(smoothed, config.canny_low, config.canny_high)
    color = lane_color_mask(bgr, config)
    color_edges = cv2.bitwise_and(edges, color)
    # Keep Canny structure even if the color prior is thin on a dashed line.
    combined = cv2.bitwise_or(color_edges, cv2.bitwise_and(edges, edges))
    # Prefer pixels that are either strong edges or lane-colored.
    prior = cv2.dilate(color, np.ones((3, 3), np.uint8), iterations=1)
    fused = cv2.bitwise_or(combined, cv2.bitwise_and(edges, prior))
    return apply_roi(fused, config)
