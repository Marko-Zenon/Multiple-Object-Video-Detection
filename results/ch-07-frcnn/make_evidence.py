"""Regenerate the CH-07 comparison and selected-checkpoint failure figures.

Run from the repository root: python results/ch-07-frcnn/make_evidence.py
Add --tables-only to regenerate the tables without local checkpoints or a GPU.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.ops import box_convert, box_iou

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from movd import data
from movd.tv_train import build_model, make_loader, predict_coco, validation_frames

RUNS = ("frcnn_baseline", "frcnn_rpn_pre3000", "frcnn_roi_pos50")
CATEGORIES = data.CLASS_NAMES
MATCH_IOU = 0.5


def read(run, artifact):
    return json.loads(
        (HERE.parent / run / f"{artifact}.json").read_text(encoding="utf-8")
    )


def write_table(name, rows):
    with (HERE / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare():
    rows = []
    for run in RUNS:
        curve = read(run, "curve")
        full = read(run, "metrics")["val_full"]
        runtime = read(run, "runtime")
        assert runtime["images_seen"] == 16800 and len(curve) == 6
        assert full["num_images"] == 2846 and full["max_dets"] == 500
        rows.append(
            {
                "run": run,
                **{
                    f"val_fast_{key}": value
                    for key, value in curve[-1]["val_fast"].items()
                },
                **{f"val_full_{key}": value for key, value in full["metrics"].items()},
                "fps": read(run, "speed")["inference"]["fps"],
                "seconds": runtime["this_process_seconds"],
            }
        )
    selected = max(rows, key=lambda row: row["val_fast_mAP"])["run"]
    for row in rows:
        row["selected"] = row["run"] == selected
        row["val_full_mAP_delta"] = row["val_full_mAP"] - rows[0]["val_full_mAP"]
    write_table("comparison.csv", rows)
    return selected


def box_xyxy(records):
    return box_convert(
        torch.tensor([row["bbox"] for row in records]).reshape(-1, 4), "xywh", "xyxy"
    )


def find_cases(frames, predictions, index):
    """Inspect individual GT overlaps for illustrations. AP comes from CH-03."""
    by_frame = defaultdict(list)
    for prediction in predictions:
        if prediction["score"] >= 0.1:
            by_frame[prediction["image_id"]].append(prediction)
    cases = {}
    for image in frames.images:
        image_id = image["id"]
        entry = index[image_id]
        targets = entry["targets"]
        detections = sorted(by_frame[image_id], key=lambda item: -item["score"])
        overlaps = box_iou(box_xyxy(targets), box_xyxy(detections))
        for target, ious in zip(targets, overlaps, strict=True):
            category = target["category_id"]
            if any(
                iou >= MATCH_IOU and det["category_id"] == category
                for iou, det in zip(ious, detections, strict=True)
            ):
                continue
            _, _, width, height = target["bbox"]
            confusion = next(
                (
                    det
                    for iou, det in zip(ious, detections, strict=True)
                    if iou >= MATCH_IOU and {category, det["category_id"]} == {3, 4}
                ),
                None,
            )
            if confusion and "car_van_confusion" not in cases:
                kind = "car_van_confusion"
            elif min(width, height) < 16 and "tiny_target_miss" not in cases:
                kind = "tiny_target_miss"
            elif len(targets) >= 75 and "dense_frame_miss" not in cases:
                kind = "dense_frame_miss"
            else:
                continue
            best = (
                detections[int(ious.argmax())]
                if detections and ious.max() > 0
                else None
            )
            if kind == "car_van_confusion":
                best = confusion
            cases[kind] = image, target, best, entry
            break
        if len(cases) == 3:
            break
    return cases


def draw_case(kind, case):
    image, target, best, entry = case
    from PIL import Image

    x, y, width, height = target["bbox"]
    pad = max(60, 2 * width, 2 * height)
    with Image.open(data.image_path("val", image)) as photo:
        figure, axis = plt.subplots(figsize=(8, 6))
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
    label = CATEGORIES[target["category_id"] - 1]
    message = "No overlapping prediction with score ≥ 0.10"
    if best:
        box = best["bbox"]
        axis.add_patch(
            Rectangle(box[:2], *box[2:], fill=False, edgecolor="red", linewidth=2)
        )
        pred_label = CATEGORIES[best["category_id"] - 1]
        iou = box_iou(box_xyxy([target]), box_xyxy([best])).item()
        message = f"Red: {pred_label}, score {best['score']:.2f}, IoU {iou:.2f}"
    axis.set(
        xlim=(max(0, x - pad), min(image["width"], x + width + pad)),
        ylim=(min(image["height"], y + height + pad), max(0, y - pad)),
    )
    axis.set_title(
        f"{kind.replace('_', ' ').title()} | {image['file_name']}\nGreen: {label} ({width:.0f}×{height:.0f} px); {message}",
        fontsize=10,
    )
    axis.axis("off")
    figure.savefig(HERE / "figures" / f"{kind}.png", bbox_inches="tight", dpi=145)
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
    }


def failures(run):
    config = json.loads(
        (HERE.parents[1] / "configs" / "runs" / f"{run}.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint = torch.load(
        Path("outputs") / run / "epoch_0006.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["config"] == config
    model = build_model(config, initialize_weights=False).cuda()
    model.load_state_dict(checkpoint["model"])
    frames = validation_frames(config, "val_fast")
    predictions = predict_coco(
        model, make_loader(frames, config), config["family"], config["amp"]
    )
    entries = data.split_annotations(data.load_coco("val"))
    cases = find_cases(frames, predictions, entries)
    if len(cases) != 3:
        raise RuntimeError(f"Expected three kinds of failure; found {list(cases)}")
    (HERE / "figures").mkdir(exist_ok=True)
    write_table(
        "failure_cases.csv", [draw_case(kind, case) for kind, case in cases.items()]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    selected = compare()
    print(f"Selected by final val_fast mAP: {selected}", flush=True)
    if not args.tables_only:
        failures(selected)


if __name__ == "__main__":
    main()
