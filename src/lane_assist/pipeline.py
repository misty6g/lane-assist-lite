"""End-to-end frame / image / video lane-assist pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lane_assist.config import PipelineConfig
from lane_assist.detect import estimate_from_fits, hough_lines, measure_fits
from lane_assist.geometry import LaneEstimate, blend_fit
from lane_assist.overlay import draw_lanes
from lane_assist.preprocess import edge_map

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class FrameResult:
    annotated: np.ndarray
    estimate: LaneEstimate
    edges: np.ndarray


class LaneAssistPipeline:
    """Stateful detector. Video calls reuse EMA-smoothed lane coefficients."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._left_fit: tuple[float, float] | None = None
        self._right_fit: tuple[float, float] | None = None

    def reset(self) -> None:
        self._left_fit = None
        self._right_fit = None

    def process_frame(self, frame: np.ndarray, *, smooth: bool = True) -> FrameResult:
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame")
        edges = edge_map(frame, self.config)
        segments = hough_lines(edges, self.config)
        measured_left, measured_right, n_left, n_right = measure_fits(
            segments, frame.shape[1], self.config
        )
        if smooth:
            self._left_fit = blend_fit(self._left_fit, measured_left, self.config.ema_alpha)
            self._right_fit = blend_fit(self._right_fit, measured_right, self.config.ema_alpha)
        else:
            self._left_fit = measured_left
            self._right_fit = measured_right
        estimate = estimate_from_fits(
            self._left_fit,
            self._right_fit,
            frame.shape,
            self.config,
            n_left,
            n_right,
        )
        annotated = draw_lanes(frame, estimate, self.config)
        return FrameResult(annotated=annotated, estimate=estimate, edges=edges)

    def process_image(self, image: np.ndarray) -> FrameResult:
        self.reset()
        return self.process_frame(image, smooth=False)

    def process_image_path(self, path: Path | str) -> FrameResult:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return self.process_image(image)

    def process_video(
        self,
        source: Path | str,
        dest: Path | str,
        *,
        max_frames: int | None = None,
        progress: bool = False,
    ) -> list[LaneEstimate]:
        self.reset()
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dest_path), fourcc, fps, (width, height))
        estimates: list[LaneEstimate] = []
        index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = self.process_frame(frame, smooth=True)
                writer.write(result.annotated)
                estimates.append(result.estimate)
                index += 1
                if progress and index % 10 == 0:
                    print(f"processed {index} frames", flush=True)
                if max_frames is not None and index >= max_frames:
                    break
        finally:
            cap.release()
            writer.release()
        return estimates


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES
