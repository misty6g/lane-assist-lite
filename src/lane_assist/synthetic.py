"""Procedural forward-camera road scenes (no third-party footage)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _lane_xs(
    y: np.ndarray,
    height: int,
    width: int,
    offset_px: float,
    curvature: float,
    horizon_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Perspective lane x-coordinates for each image row ``y``."""
    horizon = height * horizon_ratio
    t = np.clip((y - horizon) / max(height - horizon, 1.0), 0.0, 1.0)
    # Near-field half-width grows toward the bottom of the frame.
    # Kept inside a typical dashcam ROI so both markings stay visible.
    half = (0.11 + 0.22 * t) * width
    curve = curvature * width * t * t
    center = width * 0.50 - offset_px * t + curve
    return center - half, center + half


def render_scene(
    width: int = 1280,
    height: int = 720,
    *,
    offset_px: float = 0.0,
    curvature: float = 0.0,
    lighting: str = "day",
    seed: int = 0,
    dashed_center: bool = True,
) -> np.ndarray:
    """Render a BGR road image with yellow left and white right markings."""
    del seed  # kept on the signature so callers can pin a scene
    horizon_ratio = 0.42
    horizon = int(height * horizon_ratio)

    if lighting == "dusk":
        sky_top = np.array([90, 70, 40], dtype=np.float32)
        sky_bot = np.array([160, 110, 70], dtype=np.float32)
        asphalt = np.array([48, 48, 50], dtype=np.float32)
        grass = np.array([28, 55, 32], dtype=np.float32)
    elif lighting == "night":
        sky_top = np.array([20, 12, 8], dtype=np.float32)
        sky_bot = np.array([40, 28, 18], dtype=np.float32)
        asphalt = np.array([28, 28, 30], dtype=np.float32)
        grass = np.array([16, 28, 18], dtype=np.float32)
    else:
        sky_top = np.array([210, 150, 80], dtype=np.float32)
        sky_bot = np.array([245, 210, 170], dtype=np.float32)
        asphalt = np.array([62, 62, 65], dtype=np.float32)
        grass = np.array([42, 110, 48], dtype=np.float32)

    frame = np.zeros((height, width, 3), dtype=np.float32)
    for row in range(horizon + 1):
        a = row / max(horizon, 1)
        frame[row] = sky_top * (1.0 - a) + sky_bot * a
    frame[horizon:] = grass

    ys = np.arange(horizon, height, dtype=np.float32)
    left_xs, right_xs = _lane_xs(ys, height, width, offset_px, curvature, horizon_ratio)
    road = frame.copy()
    for i, y in enumerate(ys.astype(int)):
        x0 = int(np.clip(left_xs[i] - 18, 0, width - 1))
        x1 = int(np.clip(right_xs[i] + 18, 0, width - 1))
        if x1 > x0:
            road[y, x0:x1] = asphalt
    frame = road

    left_color = (20, 200, 240) if lighting != "night" else (10, 170, 220)
    right_color = (235, 235, 235) if lighting != "night" else (200, 200, 200)
    center_color = (220, 220, 220)

    left_pts = np.stack([left_xs, ys], axis=1).astype(np.int32)
    right_pts = np.stack([right_xs, ys], axis=1).astype(np.int32)
    thickness = 10 if lighting != "night" else 12
    cv2.polylines(frame, [left_pts], False, left_color, thickness, cv2.LINE_AA)
    cv2.polylines(frame, [right_pts], False, right_color, thickness, cv2.LINE_AA)

    if dashed_center:
        mid_xs = (left_xs + right_xs) / 2.0
        dash_on = True
        run = 0
        for i in range(0, len(ys) - 1):
            run += 1
            if run > 18:
                dash_on = not dash_on
                run = 0
            if not dash_on:
                continue
            p1 = (int(mid_xs[i]), int(ys[i]))
            p2 = (int(mid_xs[i + 1]), int(ys[i + 1]))
            cv2.line(frame, p1, p2, center_color, 5, cv2.LINE_AA)

    # Vehicle hood so the ROI has a natural near-field cutoff.
    hood = np.array(
        [
            [0, height],
            [0, int(height * 0.93)],
            [int(width * 0.18), int(height * 0.88)],
            [int(width * 0.82), int(height * 0.88)],
            [width, int(height * 0.93)],
            [width, height],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, hood, (22, 22, 24))

    if lighting == "night":
        # Headlight bloom on the asphalt, still leaving lane paint visible.
        glow = np.zeros_like(frame)
        cv2.ellipse(
            glow,
            (width // 2, int(height * 0.78)),
            (int(width * 0.35), int(height * 0.16)),
            0,
            0,
            360,
            (70, 70, 55),
            -1,
        )
        frame = np.clip(frame + glow * 0.45, 0, 255)

    return frame.astype(np.uint8)


def write_sample_assets(output_dir: Path | str, *, include_video: bool = True) -> dict[str, Path]:
    """Write committed-quality demo frames (and an optional short clip)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    scenes = {
        "straight": dict(offset_px=0.0, curvature=0.0, lighting="day", seed=1),
        "curve": dict(offset_px=22.0, curvature=0.10, lighting="day", seed=2),
        "dusk": dict(offset_px=-16.0, curvature=-0.04, lighting="dusk", seed=3),
        "night": dict(offset_px=10.0, curvature=0.03, lighting="night", seed=4),
    }
    for name, kwargs in scenes.items():
        path = out / f"{name}.png"
        cv2.imwrite(str(path), render_scene(**kwargs), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        written[name] = path

    if include_video:
        clip = out / "clip.mp4"
        width, height, fps, n_frames = 960, 540, 16, 48
        writer = cv2.VideoWriter(
            str(clip), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        for i in range(n_frames):
            t = i / max(n_frames - 1, 1)
            offset = 26.0 * np.sin(2 * np.pi * t)
            curve = 0.06 * np.sin(2 * np.pi * t + 0.4)
            frame = render_scene(
                width,
                height,
                offset_px=float(offset),
                curvature=float(curve),
                lighting="day",
                seed=10 + i,
            )
            writer.write(frame)
        writer.release()
        written["clip"] = clip
    return written
