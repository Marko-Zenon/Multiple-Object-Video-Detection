"""Tests for movd/submission.py and the CH-05 CLIs on synthetic predictions (no dataset or GPU needed)."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys

import pytest
import torch

from movd import data, tv_train
from movd import submission as sub

# Two resolutions that occur in test_images.csv (configs/protocol.json "test_image_sizes").
IMAGES = [
    {"id": 1, "key": "clip_001_0000001", "width": 1360, "height": 764},
    {"id": 2, "key": "clip_001_0000002", "width": 1360, "height": 764},
    {"id": 3, "key": "clip_002_0000001", "width": 680, "height": 382},
]
TEST_SIZES = [
    (1360, 764),
    (1920, 1080),
    (1344, 756),
    (680, 382),
    (1904, 1072),
    (960, 540),
]


def det(image_id, category_id, bbox, score):
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "score": score,
    }


def groups(rows, index):
    return sub.parse_prediction_string(rows[index][1])


def full_csv(rows):
    return [sub.HEADER] + [list(r) for r in rows]


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, data.REPO_ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- writer


def test_header_and_row_order_follow_the_image_table(tmp_path):
    preds = [det(3, 1, [1, 1, 5, 5], 0.9), det(1, 2, [2, 2, 5, 5], 0.8)]
    rows, stats = sub.build_rows(preds, IMAGES)
    path = tmp_path / "s.csv"
    sub.write_csv(path, rows)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "image_id,PredictionString"
    assert [line.split(",")[0] for line in lines[1:]] == [im["key"] for im in IMAGES]
    assert stats["rows"] == 3 and stats["empty_rows"] == 1


def test_class_is_category_minus_one_and_label_shift_raises():
    rows, _ = sub.build_rows(
        [det(1, c, [0, 0, 4, 4], 0.5) for c in data.CATEGORY_IDS], IMAGES
    )
    assert sorted(g[0] for g in groups(rows, 0)) == list(range(8))
    with pytest.raises(ValueError, match="category_id"):
        sub.build_rows([det(1, 0, [0, 0, 4, 4], 0.5)], IMAGES)
    with pytest.raises(ValueError, match="category_id"):
        sub.build_rows([det(1, 9, [0, 0, 4, 4], 0.5)], IMAGES)


def test_box_is_top_left_xywh_in_original_pixels():
    rows, _ = sub.build_rows(
        [det(1, 3, [100.123, 200.456, 30.5, 40.25], 0.876543214)], IMAGES
    )
    assert rows[0][1] == "2 0.87654321 100.12300 200.45600 30.50000 40.25000"


def test_boxes_are_clipped_to_the_image():
    preds = [
        det(3, 1, [-10, -5, 30, 20], 0.9),  # crosses the top-left corner
        det(
            3, 1, [670, 370, 50, 50], 0.8
        ),  # crosses the bottom-right corner of 680x382
    ]
    rows, _ = sub.build_rows(preds, IMAGES)
    assert groups(rows, 2) == [
        (0, 0.9, 0.0, 0.0, 20.0, 15.0),
        (0, 0.8, 670.0, 370.0, 10.0, 12.0),
    ]
    problems, _ = sub.validate_rows(full_csv(rows), IMAGES)
    assert problems == []


@pytest.mark.parametrize(
    "bbox, score, reason",
    [
        ([math.nan, 0, 5, 5], 0.5, "rejected_bad_box"),
        ([0, 0, math.inf, 5], 0.5, "rejected_bad_box"),
        ([0, 0, -3, 5], 0.5, "rejected_bad_box"),
        ([0, 0, 0, 5], 0.5, "rejected_empty_box"),
        (
            [0, 0, 0.000004, 5],
            0.5,
            "rejected_empty_box",
        ),  # zero width after rounding to 1e-5 px
        (
            [2000, 10, 5, 5],
            0.5,
            "rejected_empty_box",
        ),  # fully outside: zero area after clipping
        ([-50, 10, 20, 5], 0.5, "rejected_empty_box"),
        ([0, 0, 5, 5], math.nan, "rejected_bad_score"),
        ([0, 0, 5, 5], -0.1, "rejected_bad_score"),
        ([0, 0, 5, 5], 1.5, "rejected_bad_score"),
    ],
)
def test_invalid_detections_are_rejected_and_counted(bbox, score, reason):
    rows, stats = sub.build_rows(
        [det(1, 1, bbox, score), det(1, 2, [0, 0, 5, 5], 0.4)], IMAGES
    )
    assert stats[reason] == 1 and stats["written"] == 1
    assert [g[0] for g in groups(rows, 0)] == [1]


def test_images_without_valid_detections_get_an_empty_string():
    rows, stats = sub.build_rows([det(2, 1, [0, 0, 0, 0], 0.9)], IMAGES)
    assert [r[1] for r in rows] == ["", "", ""]
    assert stats["empty_rows"] == 3
    rows, _ = sub.build_rows([], IMAGES)
    assert sub.validate_rows(full_csv(rows), IMAGES)[0] == []


def test_order_is_by_score_and_independent_of_input_order():
    preds = [
        det(1, 5, [5, 5, 5, 5], 0.5),
        det(1, 2, [1, 1, 5, 5], 0.9),
        det(1, 1, [9, 9, 5, 5], 0.5),  # same score as the first: class breaks the tie
        det(1, 1, [3, 3, 5, 5], 0.5),  # same score and class: box breaks the tie
        det(1, 3, [7, 7, 5, 5], 0.700000001),
    ]
    rows, _ = sub.build_rows(preds, IMAGES)
    expected = [
        (1, 0.9, 1, 1),
        (2, 0.7, 7, 7),
        (0, 0.5, 3, 3),
        (0, 0.5, 9, 9),
        (4, 0.5, 5, 5),
    ]
    assert [g[:4] for g in groups(rows, 0)] == expected
    assert sub.build_rows(preds[::-1], IMAGES)[0] == rows


def test_cap_keeps_the_500_highest_scores_over_all_classes():
    preds = [
        det(1, 1 + i % 8, [i % 100, i // 100, 4, 4], (i + 1) / 1001)
        for i in range(1000)
    ]
    rows, stats = sub.build_rows(preds, IMAGES)
    kept = groups(rows, 0)
    assert len(kept) == 500 and stats["dropped_over_cap"] == 500
    assert min(g[1] for g in kept) == pytest.approx(501 / 1001, abs=1e-6)
    assert {g[0] for g in kept} == set(range(8))
    rows, _ = sub.build_rows(preds, IMAGES, max_dets=10)
    assert len(groups(rows, 0)) == 10


def test_unknown_image_id_raises():
    with pytest.raises(ValueError, match="not in the image table"):
        sub.build_rows([det(99, 1, [0, 0, 4, 4], 0.5)], IMAGES)


def test_duplicate_image_table_entries_raise():
    with pytest.raises(ValueError, match="duplicate image ids"):
        sub.build_rows([], IMAGES + [IMAGES[0]])


def test_full_precision_score_is_used_for_sorting():
    preds = [det(1, 1, [0, 0, 4, 4], 0.123456781), det(1, 2, [0, 0, 4, 4], 0.123456784)]
    rows, _ = sub.build_rows(preds, IMAGES)
    assert [g[0] for g in groups(rows, 0)] == [1, 0]  # both print as 0.12345678


# ---------------------------------------------------------------- parser and round trip


def test_parse_prediction_string():
    assert sub.parse_prediction_string("") == []
    assert sub.parse_prediction_string("  ") == []
    assert sub.parse_prediction_string("7 0.5 1 2 3 4 0 1.0 0.00 0.00 1.5 2.5") == [
        (7, 0.5, 1.0, 2.0, 3.0, 4.0),
        (0, 1.0, 0.0, 0.0, 1.5, 2.5),
    ]
    for bad in (
        "1 0.5 1 2 3",
        "x 0.5 1 2 3 4",
        "1.0 0.5 1 2 3 4",
        "-1 0.5 1 2 3 4",
        "1 0.5 a 2 3 4",
    ):
        with pytest.raises(ValueError):
            sub.parse_prediction_string(bad)


def test_rows_parse_back_to_the_written_predictions():
    preds = [det(1, 4, [10.5, 20.25, 30.0, 40.75], 0.75), det(3, 8, [1, 2, 3, 4], 0.25)]
    rows, _ = sub.build_rows(preds, IMAGES)
    back = sub.rows_to_predictions(rows, IMAGES)
    assert sorted(back, key=lambda p: p["image_id"]) == preds


# ---------------------------------------------------------------- validator


def valid_rows():
    rows, _ = sub.build_rows(
        [det(1, 1, [0, 0, 10, 10], 0.9), det(3, 2, [5, 5, 10, 10], 0.5)], IMAGES
    )
    return full_csv(rows)


def test_valid_file_passes_and_summary_counts():
    problems, summary = sub.validate_rows(valid_rows(), IMAGES)
    assert problems == []
    assert (
        summary["detections"] == 2
        and summary["empty_rows"] == 1
        and summary["data_rows"] == 3
    )


@pytest.mark.parametrize(
    "mutate, problem",
    [
        (lambda r: r.__setitem__(0, ["ImageId", "PredictionString"]), "header"),
        (lambda r: r.pop(), "data rows"),
        (lambda r: r.pop(), "missing"),
        (lambda r: r.__setitem__(2, [r[1][0], ""]), "duplicate"),
        (lambda r: r.__setitem__(2, ["clip_999_0000001", ""]), "unknown"),
        (lambda r: r.append(["clip_999_0000001", ""]), "unknown"),
        (
            lambda r: (
                r.__setitem__(1, r[2]) or r.__setitem__(2, ["clip_001_0000001", ""])
            ),
            "order",
        ),
        (lambda r: r.__setitem__(1, [r[1][0], "", "extra"]), "columns"),
        (lambda r: r.__setitem__(1, [r[1][0], "0 0.5 1 2 3"]), "multiple of 6"),
        (lambda r: r.__setitem__(1, [r[1][0], "8 0.5 1 2 3 4"]), "class"),
        (lambda r: r.__setitem__(1, [r[1][0], "1.0 0.5 1 2 3 4"]), "class"),
        (lambda r: r.__setitem__(1, [r[1][0], "1 1.5 1 2 3 4"]), "score"),
        (lambda r: r.__setitem__(1, [r[1][0], "1 nan 1 2 3 4"]), "score"),
        (lambda r: r.__setitem__(1, [r[1][0], "1 0.5 inf 2 3 4"]), "non-finite"),
        (lambda r: r.__setitem__(1, [r[1][0], "1 0.5 a 2 3 4"]), "non-numeric"),
        (lambda r: r.__setitem__(1, [r[1][0], "1 0.5 1 2 0 4"]), "width/height"),
        (lambda r: r.__setitem__(1, [r[1][0], "1 0.5 -1 2 3 4"]), "outside"),
        (
            lambda r: r.__setitem__(1, [r[1][0], "1 0.5 1350 2 11 4"]),
            "outside",
        ),  # 1361 > 1360
        (
            lambda r: r.__setitem__(1, [r[1][0], " ".join(["1 0.5 1 2 3 4"] * 501)]),
            "more than 500",
        ),
    ],
)
def test_validator_rejects_malformed_files(mutate, problem):
    rows = valid_rows()
    mutate(rows)
    problems, _ = sub.validate_rows(rows, IMAGES)
    assert any(problem in p for p in problems), problems


def test_validator_reports_unsorted_rows_without_failing():
    rows = valid_rows()
    rows[1] = [rows[1][0], "1 0.2 1 2 3 4 1 0.9 1 2 3 4"]
    problems, summary = sub.validate_rows(rows, IMAGES)
    assert problems == [] and summary["rows_not_sorted_by_score"] == 1


def test_empty_file_is_invalid():
    assert sub.validate_rows([], IMAGES)[0] == ["file is empty"]


# ---------------------------------------------------------------- CLIs


def write_dataset(root):
    """A fake dataset root with only test_images.csv (what the submission CLIs read)."""
    root.mkdir()
    with open(root / "test_images.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "file_name", "clip", "frame", "width", "height"])
        for im in IMAGES:
            clip, frame = im["key"].rsplit("_", 1)
            writer.writerow(
                [
                    im["key"],
                    f"test/images/{clip}/{frame}.jpg",
                    clip,
                    int(frame),
                    im["width"],
                    im["height"],
                ]
            )


def test_make_and_validate_submission_clis(tmp_path):
    root = tmp_path / "dataset"
    write_dataset(root)
    preds_path = tmp_path / "run" / "test_predictions.json"
    preds_path.parent.mkdir()
    preds_path.write_text(
        json.dumps([det(2, 3, [10, 10, 2000, 50], 0.6)]), encoding="utf-8"
    )
    out = tmp_path / "submissions" / "x.csv"
    scripts = data.REPO_ROOT / "scripts"
    made = subprocess.run(
        [
            sys.executable,
            str(scripts / "make_submission.py"),
            str(preds_path),
            "--out",
            str(out),
            "--dataset-root",
            str(root),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert made.returncode == 0, made.stdout + made.stderr
    assert (
        out.read_text(encoding="utf-8").splitlines()[2]
        == "clip_001_0000002,2 0.60000000 10.00000 10.00000 1350.00000 50.00000"
    )
    summary = json.loads(
        (
            tmp_path / "results" / "ch-05-submission" / "x" / "submission.json"
        ).read_text()
    )
    assert summary["validator"]["valid"] and summary["writer"]["written"] == 1

    command = [
        sys.executable,
        str(scripts / "validate_submission.py"),
        str(out),
        "--dataset-root",
        str(root),
    ]
    assert subprocess.run(command, capture_output=True, check=False).returncode == 0
    out.write_text(
        out.read_text(encoding="utf-8").replace("clip_002_0000001", "clip_003_0000001"),
        encoding="utf-8",
    )
    assert subprocess.run(command, capture_output=True, check=False).returncode == 1


# ---------------------------------------------------------------- inference side


@pytest.mark.parametrize("family", ["frcnn", "retina"])
def test_model_transform_restores_original_pixels_for_every_test_resolution(family):
    """Torchvision resizes and pads inside the model; postprocess must map boxes back to original pixels."""
    model = tv_train.build_model(
        {"family": family, "pretrained": False}, initialize_weights=False
    ).eval()
    images = [torch.zeros(3, h, w) for w, h in TEST_SIZES]
    batch, _ = model.transform(images)
    padded = batch.tensors.shape[-2:]
    for h, w in batch.image_sizes:
        assert max(h, w) <= 1344 and h <= padded[0] and w <= padded[1]
    resized = [
        {
            "boxes": torch.tensor([[0.0, 0.0, w, h], [w / 4, h / 4, w / 2, h / 2]]),
            "labels": torch.tensor([1, 2]),
            "scores": torch.tensor([0.9, 0.8]),
        }
        for (h, w) in batch.image_sizes
    ]
    restored = model.transform.postprocess(
        resized, batch.image_sizes, [(h, w) for w, h in TEST_SIZES]
    )
    for out, (w, h) in zip(restored, TEST_SIZES):
        expected = torch.tensor([[0.0, 0.0, w, h], [w / 4, h / 4, w / 2, h / 2]])
        assert torch.allclose(out["boxes"], expected, atol=1e-3)


class FakeDetector(torch.nn.Module):
    """Returns fixed xyxy detections; label 0 is a valid RetinaNet class (person)."""

    def forward(self, images):
        return [
            {
                "boxes": torch.tensor([[10.0, 20.0, 40.0, 60.0]]),
                "labels": torch.tensor([0 if i else 2]),
                "scores": torch.tensor([0.75]),
            }
            for i, _ in enumerate(images)
        ]


@pytest.mark.parametrize("family", ["frcnn", "retina"])
def test_predict_matches_the_training_core_conversion(family, monkeypatch):
    predict_test = load_script("predict_test")
    loader = [
        (
            [torch.zeros(3, 8, 8), torch.zeros(3, 8, 8)],
            [{"image_id": 5}, {"image_id": 6}],
        )
    ]
    ours = predict_test.predict(
        FakeDetector(), loader, family, torch.device("cpu"), amp=False
    )
    # predict_coco is CUDA-only; make .cuda() a no-op to compare the conversions on the CPU.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    reference = tv_train.predict_coco(FakeDetector(), loader, family, amp=False)
    assert ours == reference
    assert ours[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert [p["category_id"] for p in ours] == (
        [3, 1] if family == "retina" else [2, 0]
    )
