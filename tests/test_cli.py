"""CLI smoke tests."""

import json
from pathlib import Path

import cv2

from lane_assist.cli import main
from lane_assist.synthetic import render_scene


def test_cli_help_exits_zero() -> None:
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0


def test_cli_generate_and_run_image(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    assert main(["generate", "-o", str(samples), "--skip-video"]) == 0
    assert (samples / "straight.png").exists()
    assert (samples / "curve.png").exists()

    out = tmp_path / "annotated.png"
    assert main(["run", str(samples / "straight.png"), "-o", str(out)]) == 0
    annotated = cv2.imread(str(out))
    assert annotated is not None
    assert annotated.shape[2] == 3


def test_cli_run_missing_file(tmp_path: Path) -> None:
    assert main(["run", str(tmp_path / "nope.png")]) == 2


def test_cli_run_unsupported_suffix(tmp_path: Path) -> None:
    mystery = tmp_path / "data.bin"
    mystery.write_bytes(b"not-an-image")
    assert main(["run", str(mystery)]) == 2


def test_cli_run_video_and_json_report(tmp_path: Path, capsys) -> None:
    src = tmp_path / "clip.mp4"
    dest = tmp_path / "out.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 8, (640, 360))
    for i in range(4):
        writer.write(render_scene(640, 360, seed=30 + i))
    writer.release()
    assert main(["run", str(src), "-o", str(dest), "--max-frames", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["frames"] == 3
    assert dest.exists()


def test_cli_debug_writes_edges(tmp_path: Path) -> None:
    src = tmp_path / "road.png"
    dest = tmp_path / "out.png"
    cv2.imwrite(str(src), render_scene(480, 270, seed=12))
    assert main(["run", str(src), "-o", str(dest), "--debug"]) == 0
    assert (tmp_path / "out_edges.png").exists()
