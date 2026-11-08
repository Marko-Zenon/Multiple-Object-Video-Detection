"""Derive RetinaNet anchor scales from frozen, resized train_rubric targets."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from movd import data


def summarize(root):
    coco = data.load_coco("train", root)
    ids = set(data.split_image_ids("train_rubric"))
    images = {im["id"]: im for im in coco["images"] if im["id"] in ids}
    entries = data.split_annotations(
        {
            "images": list(images.values()),
            "annotations": [
                ann for ann in coco["annotations"] if ann["image_id"] in ids
            ],
        }
    )
    widths_heights = []
    for image_id, image in images.items():
        w, h = image["width"], image["height"]
        boxes, _ = data.target_arrays(entries[image_id]["targets"], w, h)
        scale = min(756 / min(w, h), 1344 / max(w, h))
        resized = np.floor(np.array([w, h]) * scale)
        widths_heights.append((boxes[:, 2:] - boxes[:, :2]) * resized / [w, h])
    wh = np.concatenate(widths_heights)
    sides = np.sqrt(wh.prod(axis=1))
    ratios = wh[:, 1] / wh[:, 0]
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    # Align the middle P3 anchor to the median target side, rounding the base
    # to a multiple of four. Preserve the original octave spacing and count.
    base = 4 * round(float(np.median(sides)) / 2 ** (1 / 3) / 4)
    sizes = [
        [int(base * 2 ** (level + step / 3)) for step in range(3)] for level in range(5)
    ]
    defaults = np.array([0.5, 1.0, 2.0])
    nearest = np.abs(np.log(ratios[:, None] / defaults)).argmin(axis=1)
    return {
        "split": "train_rubric",
        "frames": len(images),
        "non_crowd_targets": len(wh),
        "quantiles": quantiles,
        "sqrt_area_px_quantiles": np.quantile(sides, quantiles).tolist(),
        "height_width_quantiles": np.quantile(ratios, quantiles).tolist(),
        "ratio_share_between_0_5_and_2": float(
            np.mean((ratios >= 0.5) & (ratios <= 2))
        ),
        "nearest_default_ratio_counts": np.bincount(nearest, minlength=3).tolist(),
        "anchors": {"sizes": sizes, "aspect_ratios": [defaults.tolist()] * 5},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
