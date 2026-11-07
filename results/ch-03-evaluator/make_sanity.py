"""CH-03 sanity checks of movd/evaluate.py on the real val annotations -> results/ch-03-evaluator/sanity.json.

    python results/ch-03-evaluator/make_sanity.py

1. GT-as-predictions (every non-crowd val box, score 1.0) on val_fast and val_full: expect mAP >= 0.99.
2. Ignore regions: add one score-1.0 detection exactly on every ignore region, with a class cycling over
   1-8. With the provided GT (crowd kept) mAP must not change; with the crowd annotations removed from
   the GT the same detections become false positives and mAP must drop (shows the rule is active).
3. Label-map shift inside 1..8 (category_id -> category_id % 8 + 1): mAP must collapse to about 0.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from movd import data
from movd import evaluate as ev


def gt_predictions(coco: dict) -> list[dict]:
    return [
        {
            "image_id": a["image_id"],
            "category_id": a["category_id"],
            "bbox": list(a["bbox"]),
            "score": 1.0,
        }
        for a in coco["annotations"]
        if not a.get("iscrowd", 0)
    ]


def run(name: str, predictions: list[dict], gt: dict, split: str) -> dict:
    start = time.perf_counter()
    result = ev.evaluate(predictions, gt, data.split_image_ids(split))
    seconds = time.perf_counter() - start
    print(
        f"{name:<28} {split:<9} mAP@500 {result['metrics']['mAP']:.4f}  "
        f"mAP@100 {result['metrics_maxdets100']['mAP']:.4f}  ({seconds:.1f} s)"
    )
    return {
        "split": split,
        "num_images": result["num_images"],
        "num_predictions": result["num_predictions"],
        "metrics": result["metrics"],
        "metrics_maxdets100": result["metrics_maxdets100"],
        "per_class_AP": {r["name"]: r["AP"] for r in result["per_class"]},
        "density": result["density"],
        "eval_seconds": round(seconds, 1),
    }


def main() -> None:
    coco = data.load_coco("val")
    perfect = gt_predictions(coco)
    ignore_boxes = [
        (image_id, box)
        for image_id, entry in data.split_annotations(coco).items()
        for box in entry["ignore"]
    ]
    on_regions = [
        {
            "image_id": i,
            "category_id": k % data.NUM_CLASSES + 1,
            "bbox": box,
            "score": 1.0,
        }
        for k, (i, box) in enumerate(ignore_boxes)
    ]
    no_crowd_gt = dict(
        coco, annotations=[a for a in coco["annotations"] if not a.get("iscrowd", 0)]
    )
    shifted = [
        dict(p, category_id=p["category_id"] % data.NUM_CLASSES + 1) for p in perfect
    ]

    checks = {
        "gt_as_predictions_val_fast": run(
            "GT as predictions", perfect, coco, "val_fast"
        ),
        "gt_as_predictions_val_full": run(
            "GT as predictions", perfect, coco, "val_full"
        ),
        "gt_plus_ignore_region_dets_val_fast": run(
            "+ dets on ignore regions", perfect + on_regions, coco, "val_fast"
        ),
        "control_crowd_removed_val_fast": run(
            "  same, crowd GT removed", perfect + on_regions, no_crowd_gt, "val_fast"
        ),
        "label_shift_val_fast": run("label shift by one", shifted, coco, "val_fast"),
    }
    fast = checks["gt_as_predictions_val_fast"]["metrics"]["mAP"]
    with_regions = checks["gt_plus_ignore_region_dets_val_fast"]["metrics"]["mAP"]
    control = checks["control_crowd_removed_val_fast"]["metrics"]["mAP"]
    passed = {
        "val_fast_map_ge_0.99": fast >= 0.99,
        "ignore_region_dets_not_penalized": abs(with_regions - fast) < 1e-9,
        "ignore_rule_active_control_drops": control < fast - 0.01,
        "label_shift_collapses": checks["label_shift_val_fast"]["metrics"]["mAP"]
        < 0.01,
    }
    out = {
        "description": __doc__.strip().splitlines()[0],
        "val_ignore_regions": len(ignore_boxes),
        "val_images_with_over_100_targets": sum(
            r["num_images"]
            for r in checks["gt_as_predictions_val_full"]["density"]
            if r["bin"] == ">=100"
        ),
        "checks": checks,
        "passed": passed,
    }
    (HERE / "sanity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(passed, indent=2))
    if not all(passed.values()):
        raise SystemExit("sanity check failed")


if __name__ == "__main__":
    main()
