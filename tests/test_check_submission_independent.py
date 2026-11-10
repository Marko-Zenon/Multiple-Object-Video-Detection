"""The independent CH-14 submission check (`scripts/check_submission_independent.py`) on synthetic files."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "check_submission_independent.py"
)
GOOD = "2 0.9 10.0 20.0 30.5 40.25 0 0.5 0.0 0.0 1.0 1.0"


def _files(tmp_path: Path, rows, header="image_id,PredictionString"):
    images = tmp_path / "test_images.csv"
    images.write_text(
        "image_id,file_name,clip,frame,width,height\n"
        "clip_001_0000001,test/images/clip_001/0000001.jpg,clip_001,1,100,80\n"
        "clip_001_0000002,test/images/clip_001/0000002.jpg,clip_001,2,100,80\n",
        encoding="utf-8",
    )
    sub = tmp_path / "sub.csv"
    sub.write_text(
        header + "\n" + "".join(f"{k},{v}\n" for k, v in rows), encoding="utf-8"
    )
    return images, sub


def _run(images, sub, *extra):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            str(sub),
            "--test-images",
            str(images),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, json.loads(result.stdout)


def test_valid_file_passes(tmp_path):
    images, sub = _files(
        tmp_path, [("clip_001_0000001", GOOD), ("clip_001_0000002", "")]
    )
    code, report = _run(images, sub)
    assert (
        code == 0
        and report["valid"]
        and report["groups"] == 2
        and report["empty_rows"] == 1
    )


@pytest.mark.parametrize(
    "rows, header, message",
    [
        ([("clip_001_0000001", GOOD)], None, "rows, expected"),
        ([("clip_001_0000002", ""), ("clip_001_0000001", "")], None, "order"),
        ([("clip_001_0000001", ""), ("clip_001_0000001", "")], None, "duplicate"),
        (
            [("clip_001_0000001", ""), ("clip_001_0000002", "")],
            "id,PredictionString",
            "header",
        ),
        (
            [("clip_001_0000001", "8 0.9 1.0 1.0 2.0 2.0"), ("clip_001_0000002", "")],
            None,
            "class",
        ),
        (
            [("clip_001_0000001", "1 1.5 1.0 1.0 2.0 2.0"), ("clip_001_0000002", "")],
            None,
            "score",
        ),
        (
            [("clip_001_0000001", "1 0.5 90.0 1.0 20.0 2.0"), ("clip_001_0000002", "")],
            None,
            "outside",
        ),
        (
            [("clip_001_0000001", "1 0.5 1.0 1.0 0.0 2.0"), ("clip_001_0000002", "")],
            None,
            "non-positive",
        ),
        (
            [("clip_001_0000001", "1 0.5 1.0 1.0 nan 2.0"), ("clip_001_0000002", "")],
            None,
            "non-finite",
        ),
        (
            [("clip_001_0000001", "1 0.50 1.0 1.0 2.0 2.0"), ("clip_001_0000002", "")],
            None,
            "lossless",
        ),
        (
            [("clip_001_0000001", "1 0.5 1.0 1.0 2.0"), ("clip_001_0000002", "")],
            None,
            "multiple of 6",
        ),
        (
            [
                ("clip_001_0000001", " ".join(["1 0.5 1.0 1.0 2.0 2.0"] * 501)),
                ("clip_001_0000002", ""),
            ],
            None,
            "> 500",
        ),
    ],
)
def test_defects_fail(tmp_path, rows, header, message):
    images, sub = _files(tmp_path, rows, header or "image_id,PredictionString")
    code, report = _run(images, sub)
    assert code == 1 and any(message in f for f in report["failures"]), report[
        "failures"
    ]


def test_predictions_and_checkpoint_cross_check(tmp_path):
    images, sub = _files(
        tmp_path, [("clip_001_0000001", GOOD), ("clip_001_0000002", "")]
    )
    predictions = tmp_path / "pred.json"
    predictions.write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [-1.0, -2.0, 2.0, 3.0],
                    "score": 0.5,
                },
                {
                    "image_id": 1,
                    "category_id": 3,
                    "bbox": [10.0, 20.0, 30.5, 40.25],
                    "score": 0.9,
                },
            ]
        ),
        encoding="utf-8",
    )
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"checkpoint_sha256": "abc"}), encoding="utf-8")
    code, report = _run(images, sub, "--predictions", str(predictions), "--run-json", str(run),
                        "--expected-checkpoint-sha256", "abc")  # fmt: skip
    assert code == 0 and report["rows_differing_from_predictions"] == 0
    code, report = _run(
        images, sub, "--run-json", str(run), "--expected-checkpoint-sha256", "def"
    )
    assert code == 1 and "checkpoint sha256" in report["failures"][0]
    predictions.write_text(
        json.dumps(
            [{"image_id": 2, "category_id": 1, "bbox": [1, 1, 2, 2], "score": 0.4}]
        )
    )
    code, report = _run(images, sub, "--predictions", str(predictions))
    assert code == 1 and report["rows_differing_from_predictions"] == 2
