"""Pure geometry helpers used by the detector and tests.

Image coordinates follow OpenCV convention: origin at the top-left,
x right, y down. Lateral offset is measured at the bottom of the ROI
(near the vehicle) and is **positive when the camera is right of the
lane center** (a rightward drift).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from lane_assist.config import PipelineConfig


@dataclass(frozen=True)
class Line:
    """A finite line segment in image pixels."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.x1, self.y1), (self.x2, self.y2)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))


@dataclass(frozen=True)
class LaneEstimate:
    """Per-frame perception output used by the overlay and CLI report."""

    left: Line | None
    right: Line | None
    center_offset_px: float | None
    lane_width_px: float | None
    offset_lane_widths: float | None
    confidence: float
    supporting_left: int = 0
    supporting_right: int = 0

    @property
    def both_lanes(self) -> bool:
        return self.left is not None and self.right is not None


def slope(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return dy/dx. Vertical segments use ``inf`` with the sign of dy."""
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-6:
        return float("inf") if dy >= 0 else float("-inf")
    return dy / dx


def intercept(x1: float, y1: float, slope_value: float) -> float:
    """Return b in ``y = m x + b``. Vertical lines have no finite intercept."""
    if not np.isfinite(slope_value):
        return float("nan")
    return y1 - slope_value * x1


def line_slope(line: Line) -> float:
    return slope(line.x1, line.y1, line.x2, line.y2)


def classify_side(
    slope_value: float,
    midpoint_x: float,
    image_center_x: float,
    deadzone_px: float = 0.0,
) -> str | None:
    """Assign a Hough segment to the left or right lane.

    In a forward camera view, the left marking has a negative image-space
    slope and sits left of center; the right marking is the opposite.
    Segments whose midpoint falls in the center deadzone (dashed lane)
    are ignored.
    """
    if abs(midpoint_x - image_center_x) < deadzone_px:
        return None
    if not np.isfinite(slope_value):
        return "left" if midpoint_x < image_center_x else "right"
    if slope_value < 0 and midpoint_x < image_center_x:
        return "left"
    if slope_value > 0 and midpoint_x > image_center_x:
        return "right"
    return None


def bottom_x(line: Line) -> float:
    """X of the endpoint closer to the vehicle (larger image y)."""
    return line.x1 if line.y1 >= line.y2 else line.x2


def split_left_right(
    lines: Sequence[Line],
    image_width: int,
    min_abs_slope: float,
    max_abs_slope: float,
    center_deadzone_ratio: float = 0.08,
) -> tuple[list[Line], list[Line]]:
    """Filter near-horizontal / extreme slopes and bucket by lane side."""
    left: list[Line] = []
    right: list[Line] = []
    center = image_width / 2.0
    deadzone = image_width * center_deadzone_ratio
    for line in lines:
        m = line_slope(line)
        abs_m = abs(m)
        if not np.isfinite(m):
            abs_m = max_abs_slope
        if abs_m < min_abs_slope or abs_m > max_abs_slope:
            continue
        side = classify_side(m, bottom_x(line), center, deadzone)
        if side == "left":
            left.append(line)
        elif side == "right":
            right.append(line)
    return left, right


def average_slope_intercept(lines: Sequence[Line]) -> tuple[float, float] | None:
    """Average ``(slope, intercept)`` of a bucket of segments."""
    if not lines:
        return None
    slopes: list[float] = []
    intercepts: list[float] = []
    for line in lines:
        m = line_slope(line)
        if not np.isfinite(m):
            continue
        slopes.append(m)
        intercepts.append(intercept(line.x1, line.y1, m))
    if not slopes:
        return None
    return float(np.mean(slopes)), float(np.mean(intercepts))


def extrapolate_line(
    fit: tuple[float, float],
    y_top: float,
    y_bottom: float,
    image_width: int,
) -> Line | None:
    """Turn ``y = m x + b`` into a segment spanning the ROI vertically."""
    m, b = fit
    if abs(m) < 1e-6:
        return None
    x_top = (y_top - b) / m
    x_bottom = (y_bottom - b) / m
    if not (np.isfinite(x_top) and np.isfinite(x_bottom)):
        return None
    x_top = float(np.clip(x_top, -image_width, 2 * image_width))
    x_bottom = float(np.clip(x_bottom, -image_width, 2 * image_width))
    return Line(x_bottom, y_bottom, x_top, y_top)


def x_at_y(line: Line, y: float) -> float:
    """Interpolate x on a segment at a given image y."""
    y1, y2 = line.y1, line.y2
    if abs(y2 - y1) < 1e-6:
        return (line.x1 + line.x2) / 2.0
    t = (y - y1) / (y2 - y1)
    return line.x1 + t * (line.x2 - line.x1)


def lane_metrics(
    left: Line | None,
    right: Line | None,
    image_width: int,
    sample_y: float,
) -> tuple[float | None, float | None, float | None]:
    """Return ``(offset_px, lane_width_px, offset_lane_widths)`` at ``sample_y``.

    Offset is image_center - lane_center, so a rightward camera drift
    (lane center left of the image center) is positive.
    """
    if left is None or right is None:
        return None, None, None
    left_x = x_at_y(left, sample_y)
    right_x = x_at_y(right, sample_y)
    if right_x <= left_x:
        return None, None, None
    lane_center = (left_x + right_x) / 2.0
    lane_width = right_x - left_x
    offset_px = image_width / 2.0 - lane_center
    offset_norm = offset_px / lane_width
    return float(offset_px), float(lane_width), float(offset_norm)


def estimate_confidence(n_left: int, n_right: int, both: bool) -> float:
    """Heuristic confidence from supporting Hough segments."""
    if n_left == 0 and n_right == 0:
        return 0.0
    support = min(n_left, 8) + min(n_right, 8)
    base = 0.55 if both else 0.28
    return float(min(1.0, base + 0.04 * support))


def roi_vertices(width: int, height: int, config: PipelineConfig) -> np.ndarray:
    """Trapezoid covering the forward roadway, excluding sky and hood."""
    top_y = int(height * config.roi_top_ratio)
    bottom_y = int(height * config.roi_bottom_ratio)
    top_half = int(width * config.roi_top_width_ratio / 2.0)
    bottom_half = int(width * config.roi_bottom_width_ratio / 2.0)
    cx = width // 2
    polygon = np.array(
        [
            [
                (cx - bottom_half, bottom_y),
                (cx - top_half, top_y),
                (cx + top_half, top_y),
                (cx + bottom_half, bottom_y),
            ]
        ],
        dtype=np.int32,
    )
    return polygon


def blend_fit(
    previous: tuple[float, float] | None,
    current: tuple[float, float] | None,
    alpha: float,
) -> tuple[float, float] | None:
    """Exponential moving average of ``(slope, intercept)`` pairs."""
    if current is None:
        return previous
    if previous is None:
        return current
    return (
        alpha * current[0] + (1.0 - alpha) * previous[0],
        alpha * current[1] + (1.0 - alpha) * previous[1],
    )


def lines_from_hough(raw: Iterable[np.ndarray] | None) -> list[Line]:
    """Convert OpenCV HoughP output into ``Line`` objects."""
    if raw is None:
        return []
    out: list[Line] = []
    for item in raw:
        x1, y1, x2, y2 = (float(v) for v in item.reshape(-1)[:4])
        out.append(Line(x1, y1, x2, y2))
    return out
