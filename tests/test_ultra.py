"""Tests for movd/ultra.py (CH-09): YOLO export, class map, box conversion, evaluator integration.

CPU only, synthetic data (no dataset, no model weights). The Ultralytics `Results` tests need the
pinned `ultralytics` package and are skipped without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from movd import data, ultra
from movd import evaluate as ev

CATEGORIES = [{"id": c, "name": n} for c, n in zip(data.CATEGORY_IDS, data.CLASS_NAMES)]
W, H = 320, 240

try:
    from ultralytics.engine.results import Results

    HAS_ULTRALYTICS = True
except ImportError:  # pragma: no cover
    HAS_ULTRALYTICS = False
needs_ultralytics = pytest.mark.skipif(
    not HAS_ULTRALYTICS, reason="ultralytics not installed"
)


# ---------------------------------------------------------------- synthetic dataset


def _ann(image_id, category_id, box, crowd=False):
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": box,
        "area": box[2] * box[3],
        "iscrowd": int(crowd),
        **({"ignore": 1} if crowd else {}),
    }


def make_dataset(root: Path) -> dict:
    """Tiny `train` split: frame 1 has an ignore region, frame 2 has none; one degenerate target."""
    rng = np.random.default_rng(42)
    frames = {1: "clip_a/0000001.jpg", 2: "clip_b/0000001.jpg"}
    images = []
    for image_id, name in frames.items():
        path = root / "train" / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # Smooth image so JPEG error stays small.
        gradient = (
            np.linspace(0, 200, W, dtype=np.uint8)[None, :, None]
            .repeat(H, 0)
            .repeat(3, 2)
        )
        cv2.imwrite(
            str(path), (gradient + rng.integers(0, 3, gradient.shape)).astype(np.uint8)
        )
        images.append(
            {
                "id": image_id,
                "file_name": f"images/{name}",
                "width": W,
                "height": H,
                "video": name.split("/")[0],
                "frame_id": 1,
            }
        )
    anns = [
        _ann(1, 1, [10, 10, 20, 30]),
        _ann(1, 8, [100, 100, 40, 20]),  # inside the ignore region: must stay untouched
        _ann(1, 3, [300, 200, 50, 50]),  # partly outside the image: clipped
        _ann(1, 4, [50, 50, 0, 10]),  # degenerate: dropped
        _ann(2, 7, [0, 0, W, H]),
    ]
    for category_id in (
        data.CATEGORY_IDS
    ):  # one crowd region, replicated per category like the real JSON
        anns.append(_ann(1, category_id, [80, 80, 120, 100], crowd=True))
    for n, ann in enumerate(anns, 1):
        ann["id"] = n
    coco = {"images": images, "annotations": anns, "categories": CATEGORIES}
    (root / "train" / "annotations_coco.json").write_text(
        json.dumps(coco), encoding="utf-8"
    )
    return coco


@pytest.fixture()
def exported(tmp_path):
    root = tmp_path / "dataset"
    coco = make_dataset(root)
    out = tmp_path / "yolo"
    manifest = ultra.export_split(
        "train_x", "train", [1, 2], out, root, mask=True, progress=False
    )
    return root, out, coco, manifest


# ---------------------------------------------------------------- labels and class map


def test_yolo_lines_use_category_minus_one_and_normalise():
    boxes = np.array([[0, 0, W, H], [32, 24, 64, 72]], dtype=np.float32)
    lines = ultra.yolo_lines(boxes, np.array([1, 8]), W, H)
    assert lines[0] == "0 0.500000 0.500000 1.000000 1.000000"
    assert lines[1].split()[0] == "7"
    np.testing.assert_allclose(
        [float(v) for v in lines[1].split()[1:]],
        [48 / W, 48 / H, 32 / W, 48 / H],
        atol=1e-6,
    )


def test_yolo_lines_reject_out_of_range_category():
    with pytest.raises(ValueError):
        ultra.yolo_lines(
            np.array([[0, 0, 1, 1]], dtype=np.float32), np.array([0]), W, H
        )


def test_parse_yolo_lines_round_trip():
    boxes = np.array([[10.5, 20.25, 30.75, 60.0], [0, 0, W, H]])
    categories = np.array([2, 6])
    parsed, parsed_categories = ultra.parse_yolo_lines(
        "\n".join(ultra.yolo_lines(boxes, categories, W, H)), W, H
    )
    np.testing.assert_allclose(parsed, boxes, atol=1e-3)
    assert parsed_categories.tolist() == [2, 6]


@pytest.mark.parametrize(
    "bad", ["8 0.5 0.5 0.1 0.1", "-1 0.5 0.5 0.1 0.1", "0 0.5 0.5 0.1"]
)
def test_parse_yolo_lines_rejects_bad_lines(bad):
    with pytest.raises(ValueError):
        ultra.parse_yolo_lines(bad, W, H)


# ---------------------------------------------------------------- export


def test_export_layout_labels_and_list(exported):
    _root, out, _coco, manifest = exported
    assert (out / "train_x" / "images" / "clip_a" / "0000001.jpg").is_file()
    label = (
        (out / "train_x" / "labels" / "clip_a" / "0000001.txt").read_text().splitlines()
    )
    assert [line.split()[0] for line in label] == [
        "0",
        "7",
        "2",
    ]  # crowd never a target, degenerate dropped
    assert (out / "train_x.txt").read_text().splitlines() == [
        "./train_x/images/clip_a/0000001.jpg",
        "./train_x/images/clip_b/0000001.jpg",
    ]
    assert manifest["num_labels"] == 4
    assert manifest["degenerate_targets_dropped"] == 1
    assert manifest["frames_masked"] == 1 and manifest["frames_copied"] == 1
    assert manifest["labels_per_class"] == {
        "0": 1,
        "1": 0,
        "2": 1,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 1,
        "7": 1,
    }
    # The clipped box ends at the image border.
    cx, _, w, _ = map(float, label[2].split()[1:])
    assert cx + w / 2 == pytest.approx(1.0, abs=1e-6)


def test_export_masks_ignore_but_keeps_targets(exported):
    root, out, _coco, _ = exported
    original = cv2.imread(str(root / "train" / "images" / "clip_a" / "0000001.jpg"))[
        ..., ::-1
    ].astype(int)
    masked = cv2.imread(str(out / "train_x" / "images" / "clip_a" / "0000001.jpg"))[
        ..., ::-1
    ].astype(int)
    fill = np.asarray(data.IGNORE_FILL_RGB)
    # Ignore region [80, 80, 120, 100] outside the target [100, 100, 40, 20] is filled.
    assert np.abs(masked[85:95, 85:195] - fill).mean() < 3
    # The target inside the region keeps its pixels (JPEG tolerance).
    assert np.abs(masked[102:118, 102:138] - original[102:118, 102:138]).mean() < 4
    # Outside the region the image is unchanged.
    assert np.abs(masked[:60, :60] - original[:60, :60]).mean() < 4


def test_export_without_ignore_is_byte_copy(exported):
    root, out, _, _ = exported
    a = (root / "train" / "images" / "clip_b" / "0000001.jpg").read_bytes()
    assert (out / "train_x" / "images" / "clip_b" / "0000001.jpg").read_bytes() == a


def test_export_unmasked_mode_copies_every_frame(tmp_path):
    root = tmp_path / "dataset"
    make_dataset(root)
    out = tmp_path / "yolo"
    manifest = ultra.export_split(
        "val_x", "train", [1, 2], out, root, mask=False, progress=False
    )
    assert manifest["frames_copied"] == 2
    for name in ("clip_a", "clip_b"):
        assert (out / "val_x" / "images" / name / "0000001.jpg").read_bytes() == (
            root / "train" / "images" / name / "0000001.jpg"
        ).read_bytes()
    assert (
        ultra.check_export("val_x", "train", [1, 2], out, root, masked=False)[
            "num_labels"
        ]
        == 4
    )


def test_export_is_resumable_and_does_not_rewrite(exported):
    root, out, _, _ = exported
    image = out / "train_x" / "images" / "clip_a" / "0000001.jpg"
    before = image.stat().st_mtime_ns
    again = ultra.export_split(
        "train_x", "train", [1, 2], out, root, mask=True, progress=False
    )
    assert image.stat().st_mtime_ns == before
    assert again["num_labels"] == 4
    assert not list(out.rglob("*.tmp*"))


def test_check_export_passes_and_detects_tampering(exported):
    root, out, _, _ = exported
    report = ultra.check_export("train_x", "train", [1, 2], out, root)
    assert report["label_counts_match_coco_targets"] and report["class_ids_found"] == [
        0,
        2,
        6,
        7,
    ]
    assert report["masked_frames"] == 1 and report["pixel_check_max_fill_mae"] < 3
    label = out / "train_x" / "labels" / "clip_a" / "0000001.txt"
    lines = label.read_text().splitlines()
    assert lines[1].startswith("7 ")
    lines[1] = "6" + lines[1][1:]  # motor -> bus: a class shift the check must catch
    label.write_text("\n".join(lines) + "\n")
    with pytest.raises(AssertionError):
        ultra.check_export("train_x", "train", [1, 2], out, root)


def test_check_export_detects_unmasked_image(exported):
    root, out, _, _ = exported
    (out / "train_x" / "images" / "clip_a" / "0000001.jpg").write_bytes(
        (root / "train" / "images" / "clip_a" / "0000001.jpg").read_bytes()
    )
    with pytest.raises(AssertionError, match="fill colour"):
        ultra.check_export("train_x", "train", [1, 2], out, root)


def test_data_yaml(tmp_path):
    import yaml

    spec = yaml.safe_load(
        ultra.write_data_yaml(tmp_path, "train_rubric", "val_fast").read_text()
    )
    assert spec["train"] == "train_rubric.txt" and spec["val"] == "val_fast.txt"
    assert spec["nc"] == 8 and spec["names"] == dict(enumerate(data.CLASS_NAMES))
    assert Path(spec["path"]).is_absolute()


# ---------------------------------------------------------------- adapter


def test_boxes_to_coco_converts_xyxy_and_class():
    dets = ultra.boxes_to_coco(
        7,
        np.array([[10, 20, 40, 80], [5, 5, 4, 9]]),
        np.array([0, 7]),
        np.array([0.9, 0.1]),
    )
    assert dets[0] == {
        "image_id": 7,
        "category_id": 1,
        "bbox": [10.0, 20.0, 30.0, 60.0],
        "score": pytest.approx(0.9),
    }
    assert (
        dets[1]["category_id"] == 8 and dets[1]["bbox"][2] == 0.0
    )  # inverted box clipped to zero width
    with pytest.raises(ValueError):
        ultra.boxes_to_coco(7, np.array([[0, 0, 1, 1]]), np.array([8]), np.array([0.5]))


def _results(boxes_xyxy, classes, scores, shape=(H, W)):
    tensor = torch.tensor(
        [[*b, s, c] for b, c, s in zip(boxes_xyxy, classes, scores)],
        dtype=torch.float32,
    ).reshape(-1, 6)
    return Results(
        np.zeros((*shape, 3), dtype=np.uint8),
        path="x.jpg",
        names=dict(enumerate(data.CLASS_NAMES)),
        boxes=tensor,
    )


@needs_ultralytics
def test_result_to_coco_with_real_ultralytics_results():
    dets = ultra.result_to_coco(_results([[10, 20, 40, 80]], [3], [0.75]), image_id=11)
    assert dets == [
        {
            "image_id": 11,
            "category_id": 4,
            "bbox": [10.0, 20.0, 30.0, 60.0],
            "score": pytest.approx(0.75),
        }
    ]
    assert ultra.result_to_coco(_results([], [], []), image_id=11) == []


class FakeModel:
    """Stands in for `ultralytics.YOLO`: returns fixed Results and records the predict arguments."""

    def __init__(self, per_image):
        self.per_image = per_image
        self.calls = []

    def predict(self, sources, **kwargs):
        self.calls.append((list(sources), kwargs))
        return [self.per_image[s] for s in sources]


@needs_ultralytics
def test_predict_coco_applies_shared_500_cap_and_settings():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 200, (600, 2))
    many = _results(
        np.hstack([xy, xy + 10]).tolist(),
        rng.integers(0, 8, 600).tolist(),
        rng.uniform(0, 1, 600).tolist(),
    )
    model = FakeModel({"a.jpg": many, "b.jpg": _results([[1, 1, 5, 5]], [0], [0.5])})
    dets = ultra.predict_coco(
        model,
        [(1, "a.jpg"), (2, "b.jpg")],
        imgsz=1344,
        device="cpu",
        batch=2,
        progress=False,
    )
    per_image = {i: sum(d["image_id"] == i for d in dets) for i in (1, 2)}
    assert per_image == {1: ev.HEADLINE_MAX_DETS, 2: 1}
    kept_scores = sorted((d["score"] for d in dets if d["image_id"] == 1), reverse=True)
    assert kept_scores == pytest.approx(
        sorted(many.boxes.conf.tolist(), reverse=True)[:500]
    )
    kwargs = model.calls[0][1]
    assert (
        kwargs["conf"] == 0.001
        and kwargs["max_det"] == 500
        and kwargs["imgsz"] == 1344
        and kwargs["device"] == "cpu"
    )


@needs_ultralytics
def test_gt_through_labels_and_adapter_scores_perfect_map():
    """GT -> YOLO label lines -> parsed boxes -> Ultralytics Results -> adapter -> CH-03 evaluator gives mAP 1."""
    gt_objects = {
        1: [(1, [10, 10, 30, 40]), (3, [100, 50, 60, 30]), (8, [200, 150, 15, 12])],
        2: [(5, [0, 0, 120, 90])],
    }
    anns = [_ann(i, c, b) for i, objs in gt_objects.items() for c, b in objs]
    anns += [_ann(1, c, [250, 0, 60, 60], crowd=True) for c in data.CATEGORY_IDS]
    for n, ann in enumerate(anns, 1):
        ann["id"] = n
    gt = {
        "images": [
            {"id": i, "width": W, "height": H, "file_name": f"{i}.jpg"} for i in (1, 2)
        ],
        "annotations": anns,
        "categories": CATEGORIES,
    }

    entries = data.split_annotations(gt)
    detections = []
    for image_id in (1, 2):
        boxes, categories = data.target_arrays(entries[image_id]["targets"], W, H)
        parsed, parsed_categories = ultra.parse_yolo_lines(
            "\n".join(ultra.yolo_lines(boxes, categories, W, H)), W, H
        )
        result = _results(
            parsed.tolist(), (parsed_categories - 1).tolist(), [1.0] * len(parsed)
        )
        detections += ultra.result_to_coco(result, image_id)
    # A detection inside the crowd region is ignored, not a false positive.
    detections.append(
        {"image_id": 1, "category_id": 2, "bbox": [260, 5, 20, 20], "score": 0.99}
    )
    metrics = ev.evaluate(detections, gt)["metrics"]
    assert metrics["mAP"] == pytest.approx(1.0) and metrics["AP50"] == pytest.approx(
        1.0
    )


# ---------------------------------------------------------------- training arguments


def test_train_kwargs_follow_frozen_protocol(tmp_path):
    kwargs = ultra.train_kwargs(
        {"batch_size": 4}, tmp_path / "data.yaml", tmp_path / "out", device="0"
    )
    assert kwargs["epochs"] == 6 and kwargs["imgsz"] == 1344 and kwargs["seed"] == 42
    assert (
        kwargs["save_period"] == 1
        and kwargs["val"] is False
        and kwargs["max_det"] == 500
    )
    assert kwargs["fliplr"] == 0.5
    for key in (
        "mosaic",
        "mixup",
        "cutmix",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "translate",
        "scale",
        "degrees",
        "close_mosaic",
    ):
        assert kwargs[key] == 0, key
    assert (
        ultra.train_kwargs(
            {"batch_size": 4, "overrides": {"imgsz": 1600}}, "d", tmp_path, "0"
        )["imgsz"]
        == 1600
    )


@needs_ultralytics
def test_train_kwargs_are_valid_ultralytics_arguments(tmp_path):
    from ultralytics.cfg import get_cfg

    cfg = get_cfg(
        overrides=ultra.train_kwargs(
            {"batch_size": 4}, tmp_path / "data.yaml", tmp_path, device="cpu"
        )
    )
    assert cfg.mosaic == 0 and cfg.save_period == 1


# ---------------------------------------------------------------- malformed adapter input


@pytest.mark.parametrize(
    "xyxy, classes, scores",
    [
        (
            [[0, 0, 1, 1], [0, 0, 2, 2]],
            [0],
            [0.5, 0.4],
        ),  # length mismatch would be silently truncated
        ([[0, 0, 1, 1]], [0], [float("nan")]),
        ([[0, 0, float("inf"), 1]], [0], [0.5]),
        ([[0, 0, 1, 1]], [1.5], [0.5]),
        ([[0, 0, 1, 1]], [-1], [0.5]),
    ],
)
def test_boxes_to_coco_rejects_malformed_output(xyxy, classes, scores):
    with pytest.raises(ValueError):
        ultra.boxes_to_coco(1, np.array(xyxy), np.array(classes), np.array(scores))


def test_boxes_to_coco_accepts_empty_and_float_class_ids():
    assert ultra.boxes_to_coco(1, np.zeros((0, 4)), np.zeros(0), np.zeros(0)) == []
    dets = ultra.boxes_to_coco(
        1, np.array([[1, 2, 3, 4]]), np.array([2.0]), np.array([0.3])
    )
    assert dets[0]["category_id"] == 3 and isinstance(dets[0]["category_id"], int)
    json.dumps(dets)  # JSON-serialisable (plain Python floats)


# ---------------------------------------------------------------- run configs


def _write(tmp_path, value, name="cfg.json"):
    path = tmp_path / name
    path.write_text(
        value if isinstance(value, str) else json.dumps(value), encoding="utf-8"
    )
    return path


def test_repository_run_configs_are_valid():
    paths = sorted((data.REPO_ROOT / "configs" / "runs").glob("yolo_*.json"))
    assert {p.stem for p in paths} >= {"yolo_baseline", "yolo_smoke_s", "yolo_smoke_m"}
    for path in paths:
        config = ultra.load_run_config(path)
        assert config["run_id"] == path.stem
    baseline = ultra.load_run_config(data.REPO_ROOT / "configs/runs/yolo_baseline.json")
    kwargs = ultra.train_kwargs(baseline, "d.yaml", "out", "0")
    assert kwargs["epochs"] * 2800 == 16800 and kwargs["imgsz"] == 1344
    assert "fraction" not in kwargs  # the baseline trains on all of train_rubric


@pytest.mark.parametrize(
    "change, message",
    [
        ({"batchsize": 4}, "unknown"),
        ({"run_id": "../escape"}, "directory"),
        ({"batch_size": 0}, "positive"),
        ({"batch_size": "4"}, "must be int"),
        ({"batch_size": True}, "must be int"),
        ({"imgsz": 1600}, "frozen"),
        ({"epochs": 12}, "frozen"),
        ({"smoke_fraction": 1.5}, "smoke_fraction"),
    ],
)
def test_run_config_rejects_malformed_values(tmp_path, change, message):
    config = {"run_id": "yolo_x", "weights": "w.pt", "batch_size": 4, **change}
    with pytest.raises(ValueError, match=message):
        ultra.load_run_config(_write(tmp_path, config))


def test_run_config_missing_key_and_bad_json(tmp_path):
    with pytest.raises(ValueError, match="missing 'weights'"):
        ultra.load_run_config(_write(tmp_path, {"run_id": "r", "batch_size": 2}))
    with pytest.raises(ValueError, match="invalid JSON"):
        ultra.load_run_config(_write(tmp_path, "{not json"))


def test_run_config_overrides_and_smoke_are_allowed(tmp_path):
    variant = {
        "run_id": "yolo_hr",
        "weights": "w.pt",
        "batch_size": 2,
        "overrides": {"imgsz": 1600},
    }
    assert (
        ultra.train_kwargs(
            ultra.load_run_config(_write(tmp_path, variant)), "d", "o", "0"
        )["imgsz"]
        == 1600
    )
    smoke = {
        "run_id": "s",
        "weights": "w.pt",
        "batch_size": 2,
        "epochs": 2,
        "smoke_fraction": 0.07,
    }
    assert (
        ultra.train_kwargs(
            ultra.load_run_config(_write(tmp_path, smoke)), "d", "o", "0"
        )["fraction"]
        == 0.07
    )


# ---------------------------------------------------------------- failure cases


def test_iou_xywh():
    ious = ultra.iou_xywh(
        [[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 10, 10], [20, 20, 5, 5], [0, 0, 0, 0]]
    )
    assert ious.shape == (1, 4)
    assert ious[0].tolist() == pytest.approx([1.0, 50 / 150, 0.0, 0.0])
    assert ultra.iou_xywh(np.zeros((0, 4)), [[0, 0, 1, 1]]).shape == (0, 1)


def test_find_failure_cases_picks_each_kind_once():
    def target(image_id, category_id, box):
        return {"image_id": image_id, "category_id": category_id, "bbox": box}

    def det(image_id, category_id, box, score):
        return {
            "image_id": image_id,
            "category_id": category_id,
            "bbox": box,
            "score": score,
        }

    dense = [target(3, 1, [i * 20.0, 0, 18, 18]) for i in range(80)]
    entries = {
        1: {
            "targets": [
                target(1, 3, [10, 10, 50, 40]),
                target(1, 1, [200, 200, 30, 30]),
            ],
            "ignore": [],
        },
        2: {"targets": [target(2, 1, [5, 5, 8, 20])], "ignore": []},
        3: {"targets": dense, "ignore": []},
    }
    predictions = [
        det(1, 4, [10, 10, 50, 40], 0.9),  # car labelled as van: confusion
        det(1, 1, [200, 200, 30, 30], 0.8),  # correct: not a failure
        det(2, 1, [5, 5, 8, 20], 0.05),  # below the 0.10 display threshold: tiny miss
        *[det(3, 1, d["bbox"], 0.9) for d in dense[1:]],  # first dense target missed
    ]
    images = [{"id": i} for i in (1, 2, 3)]
    cases = ultra.find_failure_cases(images, entries, predictions)
    assert set(cases) == set(ultra.FAILURE_KINDS)
    assert cases["car_van_confusion"]["prediction"]["category_id"] == 4
    assert cases["car_van_confusion"]["iou"] == pytest.approx(1.0)
    assert cases["tiny_target_miss"]["prediction"] is None
    assert cases["dense_frame_miss"]["target"] is dense[0]
    assert ultra.find_failure_cases(images, entries, []).keys() <= set(
        ultra.FAILURE_KINDS
    )
