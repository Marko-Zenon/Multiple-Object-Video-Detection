"""Verify the Ultralytics -> COCO-results adapter against the CH-03 evaluator (CH-09 validation item 2).

1. GT round trip on val_full: COCO targets -> YOLO label lines -> parsed boxes -> real Ultralytics `Results`
   objects -> `movd.ultra.result_to_coco` -> 500 cap -> `movd.evaluate`. mAP must be >= 0.99.
2. Optional `--model-preflight <weights>`: run the pretrained model **on the CPU** on a few original val
   frames through `movd.ultra.predict_coco` and check the output (category ids 1-8, boxes inside the
   original image, <= 500 per image) and that the evaluator accepts it. Never uses CUDA.

Writes results/ch-09-yolo/adapter_check.json.

    python scripts/check_ultra_adapter.py --dataset-root <dataset> [--model-preflight outputs/weights/yolo11s.pt]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only, set before torch is imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from movd import data, ultra  # noqa: E402
from movd import evaluate as ev  # noqa: E402

RESULTS = data.REPO_ROOT / "results" / "ch-09-yolo"


def gt_round_trip(root, split="val_full") -> dict:
    import torch
    from ultralytics.engine.results import Results

    gt = data.load_coco("val", root)
    images = {img["id"]: img for img in gt["images"]}
    entries = data.split_annotations(gt)
    detections = []
    for image_id in data.split_image_ids(split):
        image = images[image_id]
        w, h = image["width"], image["height"]
        boxes, categories = data.target_arrays(entries[image_id]["targets"], w, h)
        parsed, parsed_categories = ultra.parse_yolo_lines(
            "\n".join(ultra.yolo_lines(boxes, categories, w, h)), w, h
        )
        rows = (
            np.column_stack([parsed, np.ones(len(parsed)), parsed_categories - 1])
            if len(parsed)
            else np.zeros((0, 6))
        )
        result = Results(
            np.zeros((h, w, 3), dtype=np.uint8),
            path=image["file_name"],
            names=dict(enumerate(data.CLASS_NAMES)),
            boxes=torch.tensor(rows, dtype=torch.float32),
        )
        detections.extend(ultra.result_to_coco(result, image_id))
    detections = ev.cap_per_image(detections, ev.HEADLINE_MAX_DETS)
    result = ev.evaluate_split(detections, split, root)
    metrics = result["metrics"]
    assert metrics["mAP"] >= 0.99, f"GT round trip mAP {metrics['mAP']:.4f} < 0.99"
    return {
        "split": split,
        "num_detections": len(detections),
        "metrics": metrics,
        "per_class_AP": {r["name"]: r["AP"] for r in result["per_class"]},
        "passed": True,
    }


def model_preflight(weights: Path, root, imgsz: int, frames: int) -> dict:
    import torch
    from ultralytics import YOLO

    assert not torch.cuda.is_available(), "preflight must run with CUDA hidden"
    model = YOLO(str(weights))
    chosen = ultra.val_frames("val_fast", root)[:frames]
    start = time.perf_counter()
    detections = ultra.predict_coco(
        model,
        chosen,
        imgsz=imgsz,
        device="cpu",
        half=False,
        progress=False,
        classes=range(data.NUM_CLASSES),
    )
    seconds = time.perf_counter() - start
    images = {img["id"]: img for img in data.load_coco("val", root)["images"]}
    per_image = {}
    for d in detections:
        image = images[d["image_id"]]
        x, y, w, h = d["bbox"]
        assert d["category_id"] in data.CATEGORY_IDS
        assert (
            -1 <= x
            and -1 <= y
            and x + w <= image["width"] + 1
            and y + h <= image["height"] + 1
        ), f"box outside image: {d}"
        per_image[d["image_id"]] = per_image.get(d["image_id"], 0) + 1
    assert max(per_image.values(), default=0) <= ev.HEADLINE_MAX_DETS
    ev.check_predictions(detections, images)
    max_x = max((d["bbox"][0] + d["bbox"][2] for d in detections), default=0)
    return {
        "weights": weights.name,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "device": "cpu",
        "imgsz": imgsz,
        "frames": [str(p.relative_to(data.split_dir("val", root))) for _, p in chosen],
        "image_sizes": sorted(
            {(images[i]["width"], images[i]["height"]) for i, _ in chosen}
        ),
        "detections_per_image": per_image,
        "max_box_right_edge_px": max_x,
        "cpu_seconds": round(seconds, 1),
        "note": "COCO-pretrained 80-class weights, not fine-tuned, restricted to COCO classes 0-7 so the adapter accepts them: checks only the output format and original-pixel scaling, not accuracy or class meaning. Without the filter the adapter rejects class 38 (verified)",
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--model-preflight",
        type=Path,
        default=None,
        help="Pretrained weights for a CPU-only predict check",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=data.load_protocol()["rubric_budget"]["input_long_side"],
    )
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()

    import ultralytics

    report = {
        "ultralytics": ultralytics.__version__,
        "gt_round_trip": gt_round_trip(args.dataset_root),
    }
    print(
        f"[OK] GT round trip val_full mAP {report['gt_round_trip']['metrics']['mAP']:.4f}"
    )
    if args.model_preflight:
        report["model_preflight"] = model_preflight(
            args.model_preflight, args.dataset_root, args.imgsz, args.frames
        )
        print(
            f"[OK] CPU preflight: {report['model_preflight']['detections_per_image']} detections, {report['model_preflight']['cpu_seconds']} s"
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "adapter_check.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
