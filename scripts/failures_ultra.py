"""Failure-case figures for an Ultralytics run from its saved val predictions (CPU only, no model, no GPU).

    python scripts/failures_ultra.py --run-id yolo_baseline [--dataset-root <dataset>] [--split val_full]

Input: `outputs/<run_id>/val_full_predictions.json`, written by `scripts/predict_ultra.py final`.
Output: `results/<run_id>/failure_cases.csv` and `results/<run_id>/figures/<kind>.png` for the three kinds
of `movd.ultra.find_failure_cases` (the CH-07 rule, so the families are comparable). The figures show
individual frames; the written interpretation belongs to the students. AP comes only from movd.evaluate.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import data, ultra  # noqa: E402


def draw_case(kind: str, case: dict, entry: dict, path: Path, root) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    image, target, best = case["image"], case["target"], case["prediction"]
    x, y, width, height = target["bbox"]
    pad = max(60, 2 * width, 2 * height)
    figure, axis = plt.subplots(figsize=(8, 6))
    with Image.open(data.image_path("val", image, root)) as photo:
        axis.imshow(photo)
    for box in entry["ignore"]:
        axis.add_patch(Rectangle(box[:2], *box[2:], color="silver", alpha=0.25))
    for other in entry["targets"]:
        box = other["bbox"]
        axis.add_patch(
            Rectangle(box[:2], *box[2:], fill=False, edgecolor="white", alpha=0.45)
        )
    axis.add_patch(
        Rectangle((x, y), width, height, fill=False, edgecolor="lime", linewidth=2)
    )
    label = data.CLASS_NAMES[target["category_id"] - 1]
    message = f"No overlapping prediction with score >= {ultra.FAILURE_MIN_SCORE:.2f}"
    if best:
        box = best["bbox"]
        axis.add_patch(
            Rectangle(box[:2], *box[2:], fill=False, edgecolor="red", linewidth=2)
        )
        message = (
            f"Red: {data.CLASS_NAMES[best['category_id'] - 1]}, "
            f"score {best['score']:.2f}, IoU {case['iou']:.2f}"
        )
    axis.set(
        xlim=(max(0, x - pad), min(image["width"], x + width + pad)),
        ylim=(min(image["height"], y + height + pad), max(0, y - pad)),
    )
    axis.set_title(
        f"{kind.replace('_', ' ')} | {image['file_name']}\n"
        f"Green: {label} ({width:.0f}x{height:.0f} px); {message}",
        fontsize=10,
    )
    axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", dpi=145)
    plt.close(figure)
    return {
        "case": kind,
        "frame": image["file_name"],
        "target": label,
        "target_width_px": round(width, 2),
        "target_height_px": round(height, 2),
        "frame_targets": len(entry["targets"]),
        "frame_ignore_regions": len(entry["ignore"]),
        "observation": message,
        "figure": f"figures/{kind}.png",
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=("val_fast", "val_full"), default="val_full")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    args = parser.parse_args()

    results = data.REPO_ROOT / "results" / args.run_id
    predictions_path = args.predictions or (
        data.REPO_ROOT / "outputs" / args.run_id / "val_full_predictions.json"
    )
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    coco = data.load_coco("val", args.dataset_root)
    images = {img["id"]: img for img in coco["images"]}
    entries = data.split_annotations(coco)
    ordered = [images[i] for i in data.split_image_ids(args.split)]

    cases = ultra.find_failure_cases(ordered, entries, predictions)
    missing = [k for k in ultra.FAILURE_KINDS if k not in cases]
    if missing:
        print(f"warning: no example found for {missing}")
    rows = [
        draw_case(
            kind,
            case,
            entries[case["image"]["id"]],
            results / "figures" / f"{kind}.png",
            args.dataset_root,
        )
        for kind, case in cases.items()
    ]
    if rows:
        with (results / "failure_cases.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(f"{row['case']}: {row['frame']} - {row['observation']}")


if __name__ == "__main__":
    main()
