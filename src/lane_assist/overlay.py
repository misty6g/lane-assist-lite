"""Draw lane polygons, markings, and a small perception HUD."""

from __future__ import annotations

import cv2
import numpy as np

from lane_assist.config import PipelineConfig
from lane_assist.geometry import LaneEstimate, Line


def _polyline(line: Line) -> np.ndarray:
    return np.array([[line.x1, line.y1], [line.x2, line.y2]], dtype=np.int32)


def draw_lanes(frame: np.ndarray, estimate: LaneEstimate, config: PipelineConfig) -> np.ndarray:
    """Return a BGR copy of ``frame`` with lane overlay and HUD."""
    out = frame.copy()
    overlay = frame.copy()
    height, width = frame.shape[:2]

    if estimate.left is not None and estimate.right is not None:
        polygon = np.vstack(
            [
                _polyline(estimate.left),
                _polyline(estimate.right)[::-1],
            ]
        )
        cv2.fillPoly(overlay, [polygon], (80, 220, 80))
        out = cv2.addWeighted(overlay, config.overlay_alpha, out, 1.0 - config.overlay_alpha, 0)

    if estimate.left is not None:
        x1, y1, x2, y2 = estimate.left.as_int_tuple()
        cv2.line(out, (x1, y1), (x2, y2), (0, 215, 255), config.line_thickness)
    if estimate.right is not None:
        x1, y1, x2, y2 = estimate.right.as_int_tuple()
        cv2.line(out, (x1, y1), (x2, y2), (255, 255, 255), config.line_thickness)

    _draw_hud(out, estimate, width, height)
    return out


def _draw_hud(out: np.ndarray, estimate: LaneEstimate, width: int, height: int) -> None:
    bar_h = max(42, height // 16)
    cv2.rectangle(out, (0, 0), (width, bar_h), (16, 16, 16), thickness=-1)
    if estimate.both_lanes:
        status = "LOCK  both"
    elif estimate.left or estimate.right:
        status = "LOCK  partial"
    else:
        status = "LOCK  none"
    if estimate.offset_lane_widths is None:
        offset_txt = "OFFSET  n/a"
    else:
        sign = "+" if estimate.offset_lane_widths >= 0 else ""
        offset_txt = f"OFFSET  {sign}{estimate.offset_lane_widths:.2f} lw"
    text = f"LANE ASSIST LITE    {status}    {offset_txt}    CONF  {estimate.confidence:.2f}"
    font_scale = max(0.50, width / 1500.0)
    cv2.putText(
        out,
        text,
        (18, int(bar_h * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
