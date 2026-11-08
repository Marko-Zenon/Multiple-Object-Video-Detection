"""Project evaluator: standard COCO bbox metrics with pycocotools. Every headline number uses it (CH-03).

Frozen protocol (`plan.md` §4, `configs/protocol.json` "evaluation" and "ignore_policy", frozen by CH-02):

* Input: a COCO-results list `[{"image_id", "category_id" 1–8, "bbox" [x, y, w, h] original px, "score"}]`.
* Metrics: mAP = AP@IoU 0.50:0.05:0.95, AP50, AP75, AP_S/M/L (COCO area ranges), per-class AP,
  and mAP per density bin (target objects per frame).
* maxDets 500 is the headline: Kaggle scores at most 500 detections **per image**. maxDets 100 is also
  reported, only to explain why legacy v0 notebook numbers (torchvision capped at 100 per image) are not
  directly comparable. Note: pycocotools' own `maxDets` counts detections per image *and per class*, so
  on its own it would let up to 8 x 500 detections per image through. The evaluator therefore first keeps
  each image's top-N detections by score over all classes (`cap_per_image`), then runs COCOeval with
  `maxDets=[N]`, which is then never binding.
* Ignore regions, evaluation side of the CH-02 policy: the provided `annotations_coco.json` is used
  **unchanged**, including the `iscrowd=1` copies of every ignore region (one copy per category).
  pycocotools treats a crowd GT as "ignore": a detection whose overlap with it (intersection / detection
  area) reaches the IoU threshold, and that matched no regular GT, is neither a TP nor an FP. Because
  every region exists once per category, this holds for detections of every class. So detections inside
  ignored regions are not penalized, as the Kaggle Data description states. This module adds no rule of
  its own; it only keeps the crowd annotations and checks the inputs.

The hidden Kaggle scorer was not exposed, so these numbers are not claimed to equal the leaderboard.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from movd import data

HEADLINE_MAX_DETS = 500
LEGACY_MAX_DETS = 100
LEGACY_KEY = f"maxdets{LEGACY_MAX_DETS}"

# Density bins: number of target (non-crowd) GT objects in the frame, [low, high).
# Val frames hold 8–116 targets; the last bin is where a 100-detection cap starts to lose recall.
DENSITY_BINS = ((0, 20), (20, 50), (50, 100), (100, math.inf))

# Axis 3 of COCOeval's precision array is the area range: 0 all, 1 small, 2 medium, 3 large.
_AREA_INDEX = {"all": 0, "small": 1, "medium": 2, "large": 3}


# ---------------------------------------------------------------- inputs


def load_predictions(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        predictions = json.load(f)
    if not isinstance(predictions, list):
        raise TypeError(f"{path}: expected a JSON list of COCO results")
    return predictions


def check_predictions(predictions: Sequence[dict], image_ids: Iterable[int]) -> None:
    """Fail loudly on malformed predictions instead of silently scoring them as wrong.

    In particular, `category_id` must be 1–8: predictions written with the 0–7 submission/YOLO class
    (a label-map shift) are rejected rather than evaluated as the wrong classes.
    """
    known_images = set(image_ids)
    for i, p in enumerate(predictions):
        missing = {"image_id", "category_id", "bbox", "score"} - p.keys()
        if missing:
            raise ValueError(f"prediction {i}: missing keys {sorted(missing)}")
        if p["image_id"] not in known_images:
            raise ValueError(
                f"prediction {i}: image_id {p['image_id']} is not in the ground truth"
            )
        if p["category_id"] not in data.CATEGORY_IDS:
            raise ValueError(
                f"prediction {i}: category_id {p['category_id']} is not in 1..{data.NUM_CLASSES}; "
                "convert 0-7 classes with movd.data.class_to_category"
            )
        box = p["bbox"]
        if (
            len(box) != 4
            or not all(math.isfinite(v) for v in box)
            or box[2] < 0
            or box[3] < 0
        ):
            raise ValueError(
                f"prediction {i}: bbox must be finite [x, y, w, h] with w, h >= 0, got {box}"
            )
        if not math.isfinite(p["score"]):
            raise ValueError(f"prediction {i}: score must be finite, got {p['score']}")


def cap_per_image(predictions: Sequence[dict], max_dets: int) -> list[dict]:
    """Keep the `max_dets` highest-scoring detections of each image, over all classes (stable on ties)."""
    by_image = defaultdict(list)
    for p in predictions:
        by_image[p["image_id"]].append(p)
    kept = []
    for preds in by_image.values():
        kept.extend(sorted(preds, key=lambda p: -p["score"])[:max_dets])
    return kept


# ---------------------------------------------------------------- pycocotools wrappers


def _quiet(fn, *args):
    """pycocotools prints progress lines; keep the CLI output readable."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def _coco_gt(gt: dict) -> COCO:
    # COCOeval writes an "ignore" key into GT annotations, so give it copies, never the cached dict.
    coco = COCO()
    coco.dataset = {
        "images": gt["images"],
        "categories": gt["categories"],
        "annotations": [dict(a) for a in gt["annotations"]],
    }
    _quiet(coco.createIndex)
    return coco


def _coco_dt(coco_gt: COCO, predictions: Sequence[dict]) -> COCO:
    if predictions:
        return _quiet(coco_gt.loadRes, [dict(p) for p in predictions])
    coco = (
        COCO()
    )  # loadRes cannot take an empty list: no detections means every AP is 0
    coco.dataset = {
        "images": coco_gt.dataset["images"],
        "categories": coco_gt.dataset["categories"],
        "annotations": [],
    }
    _quiet(coco.createIndex)
    return coco


def _run_cocoeval(
    coco_gt: COCO, predictions: Sequence[dict], image_ids: list[int], max_dets: int
) -> COCOeval:
    E = COCOeval(
        coco_gt, _coco_dt(coco_gt, cap_per_image(predictions, max_dets)), iouType="bbox"
    )
    E.params.imgIds = image_ids
    E.params.maxDets = [max_dets]
    _quiet(E.evaluate)
    _quiet(E.accumulate)
    return E


def _subset(E: COCOeval, image_ids: Sequence[int]) -> COCOeval:
    """A copy of an evaluated COCOeval restricted to some images, accumulated again.

    COCOeval.evaluate() stores one per-image result per (category, area range, image), ordered by its
    sorted image ids, so restricting means picking those entries. This avoids re-matching detections
    per density bin. (Passing a parameter object with fewer imgIds to accumulate() does not index
    correctly.) tests/test_evaluate.py checks that this equals a fresh evaluation on the same images.
    """
    all_ids = E._paramsEval.imgIds
    position = {image_id: i for i, image_id in enumerate(all_ids)}
    keep = sorted(position[i] for i in image_ids)
    n_img, n_area = len(all_ids), len(E._paramsEval.areaRng)
    sub = copy.copy(E)
    sub.params = copy.deepcopy(E.params)
    sub.params.imgIds = [all_ids[i] for i in keep]
    sub._paramsEval = copy.deepcopy(sub.params)
    sub.evalImgs = [
        E.evalImgs[k * n_area * n_img + a * n_img + i]
        for k in range(len(E._paramsEval.catIds))
        for a in range(n_area)
        for i in keep
    ]
    _quiet(sub.accumulate)
    return sub


# ---------------------------------------------------------------- metrics from the precision array


def _mean_valid(precision: np.ndarray) -> float | None:
    """Mean over valid entries (-1 = no GT), as COCOeval.summarize does. None when nothing is valid."""
    valid = precision[precision > -1]
    return float(valid.mean()) if valid.size else None


def _iou_index(E: COCOeval, iou: float) -> int:
    return int(np.argmin(np.abs(E.params.iouThrs - iou)))


def _summary(E: COCOeval) -> dict:
    """mAP, AP50, AP75, AP_S/M/L from the accumulated precision array.

    precision has shape [IoU thresholds T, recall points R, categories K, area ranges A, maxDets M=1].
    """
    p = E.eval["precision"][..., 0]
    t50, t75, all_ = _iou_index(E, 0.50), _iou_index(E, 0.75), _AREA_INDEX["all"]
    return {
        "mAP": _mean_valid(p[:, :, :, all_]),
        "AP50": _mean_valid(p[t50, :, :, all_]),
        "AP75": _mean_valid(p[t75, :, :, all_]),
        "AP_small": _mean_valid(p[:, :, :, _AREA_INDEX["small"]]),
        "AP_medium": _mean_valid(p[:, :, :, _AREA_INDEX["medium"]]),
        "AP_large": _mean_valid(p[:, :, :, _AREA_INDEX["large"]]),
    }


def _per_class(E: COCOeval, E_legacy: COCOeval, gt_counts: Counter) -> list[dict]:
    p = E.eval["precision"][:, :, :, _AREA_INDEX["all"], 0]
    p_legacy = E_legacy.eval["precision"][:, :, :, _AREA_INDEX["all"], 0]
    t50, t75 = _iou_index(E, 0.50), _iou_index(E, 0.75)
    rows = []
    for k, category_id in enumerate(
        map(int, E.params.catIds)
    ):  # numpy ints after evaluate()
        rows.append(
            {
                "class_id": data.category_to_class(category_id),
                "category_id": category_id,
                "name": data.CLASS_NAMES[category_id - 1],
                "num_gt": gt_counts[category_id],
                "AP": _mean_valid(p[:, :, k]),
                "AP50": _mean_valid(p[t50, :, k]),
                "AP75": _mean_valid(p[t75, :, k]),
                f"AP_{LEGACY_KEY}": _mean_valid(p_legacy[:, :, k]),
            }
        )
    return rows


def _bin_label(low: float, high: float) -> str:
    return f">={low}" if math.isinf(high) else f"{low}-{int(high) - 1}"


# ---------------------------------------------------------------- public API


def evaluate(
    predictions: Sequence[dict],
    gt: dict,
    image_ids: Iterable[int] | None = None,
    density_bins: Sequence[tuple[float, float]] = DENSITY_BINS,
) -> dict:
    """Evaluate COCO-results predictions against a COCO GT dict (crowd annotations kept).

    `image_ids` restricts the evaluation to a frame split (default: every GT image). Predictions on
    other images are dropped and counted. Returns a JSON-serialisable dict (see `write_outputs`).
    """
    all_ids = [img["id"] for img in gt["images"]]
    check_predictions(predictions, all_ids)
    eval_ids = sorted(set(all_ids if image_ids is None else image_ids))
    eval_set = set(eval_ids)
    kept = [p for p in predictions if p["image_id"] in eval_set]

    targets = Counter({i: 0 for i in eval_ids})
    gt_counts = Counter()
    for ann in gt["annotations"]:
        if ann["image_id"] in eval_set and not ann.get("iscrowd", 0):
            targets[ann["image_id"]] += 1
            gt_counts[ann["category_id"]] += 1

    coco_gt = _coco_gt(gt)
    E = _run_cocoeval(coco_gt, kept, eval_ids, HEADLINE_MAX_DETS)
    E_legacy = _run_cocoeval(coco_gt, kept, eval_ids, LEGACY_MAX_DETS)

    density = []
    for low, high in density_bins:
        ids = [i for i in eval_ids if low <= targets[i] < high]
        row = {
            "bin": _bin_label(low, high),
            "num_images": len(ids),
            "num_gt": sum(targets[i] for i in ids),
            "mAP": None,
            "AP50": None,
            f"mAP_{LEGACY_KEY}": None,
        }
        if ids:
            head = _summary(_subset(E, ids))
            row.update(mAP=head["mAP"], AP50=head["AP50"])
            row[f"mAP_{LEGACY_KEY}"] = _summary(_subset(E_legacy, ids))["mAP"]
        density.append(row)

    dets_per_image = Counter(p["image_id"] for p in kept)
    return {
        "num_images": len(eval_ids),
        "num_predictions": len(kept),
        "num_predictions_outside_split": len(predictions) - len(kept),
        "images_over_headline_max_dets": sum(
            n > HEADLINE_MAX_DETS for n in dets_per_image.values()
        ),
        "max_dets": HEADLINE_MAX_DETS,
        "metrics": _summary(E),
        f"metrics_{LEGACY_KEY}": _summary(E_legacy),
        "per_class": _per_class(E, E_legacy, gt_counts),
        "density": density,
        "protocol": {
            "source": "plan.md §4 / configs/protocol.json (CH-02)",
            "iou_thresholds": "0.50:0.05:0.95",
            "area_ranges": "COCO: small < 32^2, medium 32^2-96^2, large > 96^2 px",
            "max_dets": "top-N detections per image over all classes, by score (N=500 headline, 100 legacy)",
            "ignore_regions": "provided COCO GT unchanged, iscrowd=1 copies kept (pycocotools crowd matching)",
            "density": "target (non-crowd) GT objects per frame, bins [low, high)",
            "pycocotools": version("pycocotools"),
        },
    }


def evaluate_split(
    predictions: Sequence[dict], split: str = "val_full", root: str | Path | None = None
) -> dict:
    """Evaluate on a frozen val frame split (`val_full` for headline numbers, `val_fast` for curves)."""
    if split not in ("val_full", "val_fast"):
        raise ValueError(f"evaluation splits are val_full or val_fast, got {split!r}")
    result = evaluate(
        predictions, data.load_coco("val", root), data.split_image_ids(split)
    )
    result["split"] = split
    return result


def write_outputs(result: dict, out_dir: str | Path) -> None:
    """Write `metrics.json` (everything) and `per_class.csv` (per-class rows) into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    rows = result["per_class"]
    with open(out_dir / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_summary(result: dict) -> str:
    def fmt(v):
        return "  n/a " if v is None else f"{v:.4f}"

    head, legacy = result["metrics"], result[f"metrics_{LEGACY_KEY}"]
    lines = [
        f"{'metric':<10} {'maxDets ' + str(HEADLINE_MAX_DETS):>12} {'maxDets ' + str(LEGACY_MAX_DETS):>12}"
    ]
    lines += [f"{k:<10} {fmt(head[k]):>12} {fmt(legacy[k]):>12}" for k in head]
    lines.append(
        "per class (AP @500): "
        + ", ".join(f"{r['name']} {fmt(r['AP'])}" for r in result["per_class"])
    )
    lines.append(
        "density (mAP @500): "
        + ", ".join(f"{r['bin']} {fmt(r['mAP'])}" for r in result["density"])
    )
    return "\n".join(lines)
