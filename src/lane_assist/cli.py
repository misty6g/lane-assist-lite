"""Command-line interface for Lane Assist Lite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from lane_assist.config import PipelineConfig
from lane_assist.pipeline import LaneAssistPipeline, is_image_path, is_video_path
from lane_assist.synthetic import write_sample_assets


def _estimate_dict(estimate) -> dict:
    return {
        "left": estimate.left.as_int_tuple() if estimate.left else None,
        "right": estimate.right.as_int_tuple() if estimate.right else None,
        "center_offset_px": estimate.center_offset_px,
        "lane_width_px": estimate.lane_width_px,
        "offset_lane_widths": estimate.offset_lane_widths,
        "confidence": estimate.confidence,
        "supporting_left": estimate.supporting_left,
        "supporting_right": estimate.supporting_right,
    }


def cmd_run(args: argparse.Namespace) -> int:
    source = Path(args.input)
    if not source.exists():
        print(f"error: input not found: {source}", file=sys.stderr)
        return 2
    dest = Path(args.output) if args.output else _default_output(source)
    dest.parent.mkdir(parents=True, exist_ok=True)

    pipeline = LaneAssistPipeline(PipelineConfig())
    if is_image_path(source):
        result = pipeline.process_image_path(source)
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9] if dest.suffix.lower() == ".png" else []
        if not cv2.imwrite(str(dest), result.annotated, params):
            print(f"error: failed to write {dest}", file=sys.stderr)
            return 1
        if args.debug:
            debug_path = dest.with_name(dest.stem + "_edges" + dest.suffix)
            cv2.imwrite(str(debug_path), result.edges)
        print(json.dumps({"output": str(dest), **_estimate_dict(result.estimate)}, indent=2))
        return 0
    if is_video_path(source):
        estimates = pipeline.process_video(
            source, dest, max_frames=args.max_frames, progress=args.progress
        )
        locked = sum(1 for e in estimates if e.both_lanes)
        report = {
            "output": str(dest),
            "frames": len(estimates),
            "both_lanes_frames": locked,
            "mean_offset_lane_widths": _mean(
                [e.offset_lane_widths for e in estimates if e.offset_lane_widths is not None]
            ),
            "mean_confidence": _mean([e.confidence for e in estimates]),
        }
        print(json.dumps(report, indent=2))
        return 0
    print(f"error: unsupported input type: {source.suffix}", file=sys.stderr)
    return 2


def cmd_generate(args: argparse.Namespace) -> int:
    dest = Path(args.output)
    written = write_sample_assets(dest, include_video=not args.skip_video)
    print(json.dumps({k: str(v) for k, v in written.items()}, indent=2))
    return 0


def _default_output(source: Path) -> Path:
    suffix = source.suffix.lower()
    folder = Path("output")
    return folder / f"{source.stem}_annotated{suffix}"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lane-assist",
        description="Detect lane markings in a forward-camera image or short clip.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Annotate an image or video")
    run.add_argument("input", help="Path to an image (.png/.jpg) or video (.mp4)")
    run.add_argument("-o", "--output", help="Annotated output path")
    run.add_argument("--debug", action="store_true", help="Also write the ROI edge map")
    run.add_argument("--max-frames", type=int, default=None, help="Stop video after N frames")
    run.add_argument("--progress", action="store_true", help="Print video progress")
    run.set_defaults(func=cmd_run)

    gen = sub.add_parser("generate", help="Write synthetic demo frames (and a short clip)")
    gen.add_argument("-o", "--output", default="data/samples", help="Destination directory")
    gen.add_argument("--skip-video", action="store_true", help="Write stills only")
    gen.set_defaults(func=cmd_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
