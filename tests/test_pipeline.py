"""Pipeline integration tests: images, video, and failure cases."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from lane_assist.pipeline import LaneAssistPipeline, is_image_path, is_video_path
from lane_assist.synthetic import render_scene


def test_process_image_preserves_shape_and_locks() -> None:
    frame = render_scene(960, 540, seed=3)
    result = LaneAssistPipeline().process_image(frame)
    assert result.annotated.shape == frame.shape
    assert result.edges.shape == frame.shape[:2]
    assert result.estimate.both_lanes
    assert result.estimate.left is not None
    assert result.estimate.right is not None


def test_empty_frame_raises() -> None:
    with pytest.raises(ValueError):
        LaneAssistPipeline().process_frame(np.zeros((0, 0, 3), dtype=np.uint8))


def test_missing_image_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LaneAssistPipeline().process_image_path(tmp_path / "missing.png")


def test_process_image_path_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "road.png"
    cv2.imwrite(str(src), render_scene(640, 360, seed=8))
    result = LaneAssistPipeline().process_image_path(src)
    assert result.annotated.shape[0] == 360


def test_video_writes_annotated_clip(tmp_path: Path) -> None:
    src = tmp_path / "in.mp4"
    dest = tmp_path / "out.mp4"
    width, height, fps, n = 640, 360, 8, 6
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(n):
        writer.write(render_scene(width, height, offset_px=float(i * 3), seed=20 + i))
    writer.release()

    estimates = LaneAssistPipeline().process_video(src, dest)
    assert dest.exists() and dest.stat().st_size > 0
    assert len(estimates) == n
    assert any(e.both_lanes for e in estimates)

    cap = cv2.VideoCapture(str(dest))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == (height, width, 3)


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LaneAssistPipeline().process_video(tmp_path / "nope.mp4", tmp_path / "out.mp4")


def test_path_type_helpers() -> None:
    assert is_image_path(Path("a.PNG"))
    assert is_video_path(Path("b.mp4"))
    assert not is_image_path(Path("c.mp4"))
    assert not is_video_path(Path("d.png"))


def test_smoothing_holds_previous_fit_across_a_blank_frame() -> None:
    pipeline = LaneAssistPipeline()
    road = render_scene(640, 360, seed=9)
    blank = np.zeros_like(road)
    first = pipeline.process_frame(road, smooth=True)
    held = pipeline.process_frame(blank, smooth=True)
    assert first.estimate.both_lanes
    assert held.estimate.left is not None
    assert held.estimate.right is not None
