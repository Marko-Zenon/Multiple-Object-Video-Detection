"""Tests for movd/infer.py (CH-13): tiling geometry, RT-DETR decoding, NMS/WBF merging, pipeline configs.

CPU only, except the last test, which runs the CH-11 checkpoint (git-ignored) on the GPU and is skipped
without them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from movd import data, infer

CHECKPOINT = (
    data.REPO_ROOT / "outputs" / "rtdetr_baseline" / "train" / "weights" / "last.pt"
)
FINAL_INFERENCE = data.REPO_ROOT / "configs" / "runs" / "final_inference.json"


def dets(boxes, scores, classes) -> infer.Dets:
    return infer.Dets(
        np.asarray(boxes, np.float32).reshape(-1, 4),
        np.asarray(scores, np.float32),
        np.asarray(classes, np.int64),
    )


# ---------------------------------------------------------------- views


@pytest.mark.parametrize("size", [(1904, 1071), (1344, 756), (3840, 2160)])
def test_tile_windows_cover_frame_and_keep_aspect(size):
    w, h = size
    windows = infer.tile_windows(w, h, grid=2, frac=0.6)
    assert len(windows) == 4
    covered = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in windows:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        assert abs((x1 - x0) / (y1 - y0) - w / h) < 0.01
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_tile_windows_reject_gaps():
    with pytest.raises(ValueError):
        infer.tile_windows(100, 100, grid=2, frac=0.4)


def test_crop_view_and_names():
    image = np.arange(6 * 10 * 3, dtype=np.uint8).reshape(6, 10, 3)
    flipped, window = infer.crop_view(image, "flip")
    assert window == (0, 0, 10, 6) and np.array_equal(flipped, image[:, ::-1])
    tile, (x0, y0, x1, y1) = infer.crop_view(image, "tile3")
    assert np.array_equal(tile, image[y0:y1, x0:x1]) and (x1, y1) == (10, 6)
    assert infer.view_names("tiles") == ["tile0", "tile1", "tile2", "tile3"]
    assert infer.view_names("full") == ["full"]


# ---------------------------------------------------------------- decoding


def ultralytics_reference(raw: infer.Raw, k: int, conf: float):
    """Ultralytics 8.4.207: top-k over the flattened (query, class) scores, then score > conf."""
    scores = torch.from_numpy(raw.scores)[None]
    values, index = scores.flatten(1).topk(k, dim=1)
    nc = scores.shape[-1]
    query = index // nc
    keep = values[0] > conf
    return (
        raw.boxes[query[0][keep].numpy()],
        values[0][keep].numpy(),
        (index % nc)[0][keep].numpy(),
    )


def test_decode_matches_ultralytics_topk():
    rng = np.random.default_rng(0)
    raw = infer.Raw(
        rng.uniform(0, 100, (300, 4)).astype(np.float32),
        rng.uniform(0, 1, (300, 8)).astype(np.float32),
    )
    raw.scores[:50] = 0.0005  # below conf
    for k in (100, 300, 500):
        got = infer.decode(raw, k=k, conf=0.001)
        boxes, scores, classes = ultralytics_reference(raw, k, 0.001)
        assert np.array_equal(got.boxes, boxes) and np.array_equal(got.scores, scores)
        assert np.array_equal(got.classes, classes)
        assert len(got) <= k


def test_decode_k500_allows_several_classes_per_query():
    scores = np.zeros((2, 8), np.float32)
    scores[0, [2, 3]] = [0.9, 0.8]  # one query, car and van
    raw = infer.Raw(np.array([[0, 0, 10, 10], [5, 5, 9, 9]], np.float32), scores)
    got = infer.decode(raw, k=16, conf=0.001)
    assert sorted(got.classes.tolist()) == [2, 3]
    assert np.array_equal(got.boxes[0], got.boxes[1])


# ---------------------------------------------------------------- merging


def test_nms_is_classwise():
    d = dets(
        [[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 10, 10]], [0.9, 0.8, 0.7], [0, 0, 1]
    )
    kept = infer.nms(d, 0.5)
    assert kept.scores.tolist() == pytest.approx([0.9, 0.7])
    assert kept.classes.tolist() == [0, 1]


def test_wbf_fuses_and_scales_scores():
    a = dets([[0, 0, 10, 10]], [0.8], [2])
    b = dets([[2, 0, 12, 10]], [0.4], [2])
    fused = infer.wbf([a, b], iou=0.5)
    assert len(fused) == 1
    # score-weighted mean box; mean score x min(2, 2) / 2
    assert fused.boxes[0].tolist() == pytest.approx(
        [(0.8 * 0 + 0.4 * 2) / 1.2, 0, (0.8 * 10 + 0.4 * 12) / 1.2, 10]
    )
    assert fused.scores[0] == pytest.approx(0.6)
    alone = infer.wbf([a, dets([[50, 50, 60, 60]], [0.4], [2])], iou=0.5)
    assert sorted(alone.scores.tolist()) == pytest.approx(
        [0.2, 0.4]
    )  # each seen by 1 of 2 views
    other_class = infer.wbf([a, dets([[0, 0, 10, 10]], [0.4], [3])], iou=0.5)
    assert len(other_class) == 2


def test_cap_keeps_top_scores():
    d = dets(np.zeros((600, 4)), np.linspace(0, 1, 600), np.zeros(600))
    kept = infer.cap(d, 500)
    assert len(kept) == 500 and kept.scores.min() == pytest.approx(
        np.linspace(0, 1, 600)[100]
    )


def test_combine_and_pass_specs():
    config = {
        "passes": [
            {"model": "a", "imgsz": 1344, "views": "full"},
            {"model": "b", "imgsz": 1344, "views": "tiles", "k": 500},
        ],
        "decode": {"k": 300, "conf": 0.001},
        "merge": {"method": "nms", "iou": 0.6},
        "max_det": 500,
    }
    specs = infer.pass_specs(config)
    assert [s["view"] for s in specs] == ["full", "tile0", "tile1", "tile2", "tile3"]
    assert [s["k"] for s in specs] == [300, 500, 500, 500, 500]
    rng = np.random.default_rng(1)
    raws = [
        infer.Raw(
            rng.uniform(0, 500, (300, 4)).astype(np.float32),
            rng.uniform(0, 1, (300, 8)).astype(np.float32),
        )
        for _ in specs
    ]
    for r in raws:
        r.boxes[:, 2:] = r.boxes[:, :2] + 20
    out = infer.combine(raws, config)
    assert 0 < len(out) <= 500 and np.all(np.diff(out.scores) <= 0)
    with pytest.raises(ValueError):
        infer.combine(raws[:2], config)
    with pytest.raises(ValueError):
        infer.combine(raws, {**config, "merge": None})


def test_final_inference_config_is_valid():
    if not FINAL_INFERENCE.exists():
        pytest.skip("configs/runs/final_inference.json not frozen yet")
    frozen = json.loads(FINAL_INFERENCE.read_text(encoding="utf-8"))
    config = frozen["config"]
    specs = infer.pass_specs(config)
    assert specs and config["max_det"] <= 500
    assert all(s["k"] <= 500 for s in specs)


# ---------------------------------------------------------------- live pass (GPU, needs the checkpoint)


@pytest.mark.skipif(
    not (
        CHECKPOINT.exists()
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 0
    ),
    reason="needs the CH-11 checkpoint and a visible CUDA device",
)
def test_raw_pass_reproduces_ultralytics_predictor():
    import cv2

    from movd import rtdetr, ultra

    model = rtdetr.load_model(CHECKPOINT, "rtdetr")
    for _, path in ultra.val_frames("val_fast")[:2]:
        image = cv2.imread(str(path))
        raw, result = infer.raw_pass(model, image, 1344, "0", return_result=True)
        got = infer.decode(raw, 300, 0.001)
        assert np.array_equal(got.boxes, result.boxes.xyxy.cpu().numpy())
        assert np.array_equal(got.scores, result.boxes.conf.cpu().numpy())
        assert np.array_equal(
            got.classes, result.boxes.cls.cpu().numpy().astype(np.int64)
        )
