"""Tests for movd/evaluate.py and movd/speed.py (CH-03) on small synthetic COCO data (no dataset needed)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from movd import data
from movd import evaluate as ev
from movd.speed import benchmark

REPO_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = [{"id": c, "name": n} for c, n in zip(data.CATEGORY_IDS, data.CLASS_NAMES)]


def make_gt(
    objects: dict[int, list[tuple[int, list[float]]]],
    crowds: dict[int, list[list[float]]] | None = None,
) -> dict:
    """COCO GT. `objects[image_id] = [(category_id, [x, y, w, h]), ...]`.

    Each crowd box is written once per category with iscrowd=1, ignore=1, like the provided val JSON.
    """
    crowds = crowds or {}
    images = [
        {"id": i, "width": 2000, "height": 2000, "file_name": f"{i}.jpg"}
        for i in sorted(set(objects) | set(crowds))
    ]
    anns = []
    for image_id, objs in objects.items():
        for category_id, box in objs:
            anns.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": box,
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                }
            )
    for image_id, boxes in crowds.items():
        for box in boxes:
            for category_id in data.CATEGORY_IDS:
                anns.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": box,
                        "area": box[2] * box[3],
                        "iscrowd": 1,
                        "ignore": 1,
                    }
                )
    for n, ann in enumerate(anns, 1):
        ann["id"] = n
    return {"images": images, "categories": CATEGORIES, "annotations": anns}


def perfect(gt: dict, score: float = 0.9) -> list[dict]:
    return [
        {
            "image_id": a["image_id"],
            "category_id": a["category_id"],
            "bbox": list(a["bbox"]),
            "score": score,
        }
        for a in gt["annotations"]
        if not a["iscrowd"]
    ]


def grid(
    n: int, category_id: int = 3, size: float = 40
) -> list[tuple[int, list[float]]]:
    """n non-overlapping boxes."""
    return [
        (category_id, [(k % 40) * 50.0, (k // 40) * 50.0, size, size]) for k in range(n)
    ]


# Two images, several classes, every COCO area range (small < 32^2 <= medium < 96^2 <= large).
SCENE = {
    1: [(1, [10, 10, 20, 20]), (3, [100, 100, 50, 50]), (7, [300, 300, 150, 120])],
    2: [(3, [10, 10, 60, 40]), (8, [500, 500, 25, 25]), (2, [800, 800, 100, 100])],
}


def test_perfect_predictions_give_map_one():
    gt = make_gt(SCENE)
    result = ev.evaluate(perfect(gt), gt)
    for name, value in result["metrics"].items():
        assert value == pytest.approx(1.0), name
    by_name = {r["name"]: r for r in result["per_class"]}
    for name in ("person", "bicycle", "car", "bus", "motor"):
        assert by_name[name]["AP"] == pytest.approx(1.0)
    assert (
        by_name["truck"]["AP"] is None and by_name["truck"]["num_gt"] == 0
    )  # no GT -> undefined, not 0


def test_false_positive_lowers_map():
    gt = make_gt(SCENE)
    preds = perfect(gt) + [
        {"image_id": 1, "category_id": 3, "bbox": [1500, 1500, 50, 50], "score": 0.95}
    ]
    assert ev.evaluate(preds, gt)["metrics"]["mAP"] < 1.0


def test_detection_inside_ignore_region_is_not_penalized():
    region = [1000, 1000, 400, 300]
    inside = [  # high-scoring detections of two different classes, fully inside the crowd region
        {"image_id": 1, "category_id": 3, "bbox": [1050, 1050, 60, 40], "score": 0.99},
        {"image_id": 1, "category_id": 1, "bbox": [1200, 1100, 20, 30], "score": 0.98},
    ]
    with_region = make_gt(SCENE, crowds={1: [region]})
    assert ev.evaluate(perfect(with_region) + inside, with_region)["metrics"][
        "mAP"
    ] == pytest.approx(1.0)

    # Control: the same detections without the crowd annotation are false positives.
    without_region = make_gt(SCENE)
    assert (
        ev.evaluate(perfect(without_region) + inside, without_region)["metrics"]["mAP"]
        < 1.0
    )


def test_detection_mostly_outside_ignore_region_is_still_penalized():
    gt = make_gt(SCENE, crowds={1: [[1000, 1000, 100, 100]]})
    # Only a quarter of this box lies in the region: intersection / detection area = 0.25 < 0.5.
    straddling = {
        "image_id": 1,
        "category_id": 3,
        "bbox": [1050, 1050, 100, 100],
        "score": 0.99,
    }
    assert ev.evaluate(perfect(gt) + [straddling], gt)["metrics"]["mAP"] < 1.0


def test_more_than_100_correct_detections_in_one_image_all_count():
    # 150 objects of 3 classes (50 each): pycocotools' per-class maxDets=100 alone would never bind here.
    objs = [(k % 3 + 1, box) for k, (_, box) in enumerate(grid(150))]
    gt = make_gt({1: objs})
    result = ev.evaluate(perfect(gt), gt)
    assert result["max_dets"] == ev.HEADLINE_MAX_DETS == 500
    assert result["metrics"]["mAP"] == pytest.approx(1.0)
    # The legacy per-image cap keeps only 100 of 150 detections, so recall is at most 2/3 and AP drops.
    assert result["metrics_maxdets100"]["mAP"] < 0.7
    dense = next(r for r in result["density"] if r["bin"] == ">=100")
    assert dense["num_images"] == 1 and dense["mAP"] == pytest.approx(1.0)


def test_cap_per_image_keeps_top_scores_over_all_classes():
    preds = [
        {"image_id": 1, "category_id": c, "bbox": [0, 0, 1, 1], "score": s}
        for c, s in [(1, 0.2), (2, 0.9), (3, 0.5), (1, 0.7)]
    ]
    preds.append({"image_id": 2, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.1})
    kept = ev.cap_per_image(preds, 2)
    assert sorted((p["image_id"], p["score"]) for p in kept) == [
        (1, 0.7),
        (1, 0.9),
        (2, 0.1),
    ]


def test_detections_beyond_500_per_image_are_counted():
    gt = make_gt({1: grid(10)})
    noise = [
        {
            "image_id": 1,
            "category_id": 3,
            "bbox": [1800.0, 1800.0, 10.0, 10.0],
            "score": 0.01,
        }
    ] * 600
    result = ev.evaluate(perfect(gt) + noise, gt)
    assert result["images_over_headline_max_dets"] == 1


def test_zero_based_class_ids_are_rejected():
    gt = make_gt(SCENE)
    shifted = [
        dict(p, category_id=p["category_id"] - 1) for p in perfect(gt)
    ]  # 0-7 submission classes
    with pytest.raises(ValueError, match="category_id"):
        ev.evaluate(shifted, gt)


def test_in_range_class_shift_scores_zero():
    gt = make_gt(SCENE)
    # Shift by one inside 1..8 (wrong label map): every box lands on the wrong class.
    shifted = [
        dict(p, category_id=p["category_id"] % data.NUM_CLASSES + 1)
        for p in perfect(gt)
    ]
    result = ev.evaluate(shifted, gt)
    assert result["metrics"]["mAP"] == pytest.approx(0.0)
    # The label map itself round-trips: class = category_id - 1.
    restored = [
        dict(
            p,
            category_id=data.class_to_category(
                data.category_to_class(p["category_id"])
            ),
        )
        for p in perfect(gt)
    ]
    assert ev.evaluate(restored, gt)["metrics"]["mAP"] == pytest.approx(1.0)


def test_density_bins_match_a_fresh_evaluation_on_the_same_images():
    objects = {
        1: grid(5),
        2: grid(30, category_id=1),
        3: grid(60, category_id=4),
        4: grid(120),
    }
    gt = make_gt(objects)
    preds = perfect(gt)
    preds = [
        dict(p, score=0.5 + 0.4 * ((k * 7) % 10) / 10) for k, p in enumerate(preds)
    ]
    for k, p in enumerate(preds):  # perturb some boxes so APs differ from 1.0
        if k % 3 == 0:
            p["bbox"] = [p["bbox"][0] + 8, p["bbox"][1] + 8, p["bbox"][2], p["bbox"][3]]
    preds += [
        {"image_id": i, "category_id": 3, "bbox": [1900, 1900, 30, 30], "score": 0.95}
        for i in objects
    ]
    result = ev.evaluate(preds, gt)
    bins = {r["bin"]: r for r in result["density"]}
    assert [bins[b]["num_images"] for b in ("0-19", "20-49", "50-99", ">=100")] == [
        1,
        1,
        1,
        1,
    ]
    for image_id, label in [(1, "0-19"), (2, "20-49"), (3, "50-99"), (4, ">=100")]:
        fresh = ev.evaluate(preds, gt, image_ids=[image_id])
        assert bins[label]["mAP"] == pytest.approx(fresh["metrics"]["mAP"])
        assert bins[label]["AP50"] == pytest.approx(fresh["metrics"]["AP50"])


def test_split_restriction_drops_other_images():
    gt = make_gt(SCENE)
    result = ev.evaluate(perfect(gt), gt, image_ids=[1])
    assert result["num_images"] == 1
    assert (
        result["num_predictions"] == 3 and result["num_predictions_outside_split"] == 3
    )
    assert result["metrics"]["mAP"] == pytest.approx(1.0)


def test_empty_predictions_score_zero():
    gt = make_gt(SCENE)
    assert ev.evaluate([], gt)["metrics"]["mAP"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"image_id": 99, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.5},
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, -1, 1], "score": 0.5},
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 1], "score": 0.5},
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]},
    ],
)
def test_malformed_predictions_are_rejected(bad):
    with pytest.raises(ValueError):
        ev.evaluate([bad], make_gt(SCENE))


def test_gt_dict_is_not_mutated():
    gt = make_gt(SCENE, crowds={1: [[1000, 1000, 100, 100]]})
    before = json.dumps(gt, sort_keys=True)
    ev.evaluate(perfect(gt), gt)
    assert json.dumps(gt, sort_keys=True) == before


def test_write_outputs(tmp_path):
    gt = make_gt(SCENE)
    ev.write_outputs(ev.evaluate(perfect(gt), gt), tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["metrics"]["mAP"] == pytest.approx(1.0)
    lines = (tmp_path / "per_class.csv").read_text().splitlines()
    assert (
        lines[0].startswith("class_id,category_id,name,num_gt,AP")
        and len(lines) == 1 + data.NUM_CLASSES
    )


def test_cli_help_documents_usage():
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for text in ("--split", "--out", "val_full", "maxDets 500"):
        assert text in out


# ---------------------------------------------------------------- speed benchmark


def test_benchmark_times_only_the_timed_calls():
    calls = []
    result = benchmark(
        lambda batch: calls.append(batch),
        [torch.zeros(1)] * 4,
        "cpu",
        batch_size=2,
        warmup=3,
        input_size="1344x756",
    )
    assert len(calls) == 3 + 4
    assert (
        result["timed_batches"] == 4
        and result["warmup_batches"] == 3
        and result["batch_size"] == 2
    )
    assert result["fps"] > 0 and result["latency_ms_per_image"] > 0
    assert result["input_size"] == "1344x756" and result["device"] == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_benchmark_records_gpu():
    x = torch.randn(256, 256, device="cuda")
    result = benchmark(lambda b: b @ b, [x] * 3, "cuda", warmup=2)
    assert result["gpu"] == torch.cuda.get_device_name(0)
