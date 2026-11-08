"""Val round-trip check of the submission format (CH-05, plan.md CH-05 scope 4).

    python scripts/submission_roundtrip.py --synthetic --name synthetic_seed42       # CPU, no model
    python scripts/submission_roundtrip.py --predictions <val COCO-results.json> --name <run_id>

Steps: val_full predictions -> submission CSV (same writer as the test submission, key = COCO image id)
-> validate -> parse back -> evaluate with `movd.evaluate.evaluate_split`, and compare with evaluating the
input predictions directly. The acceptance threshold is |difference| <= 1e-6 for every headline metric.

`--synthetic` builds seeded predictions from the val GT (jittered true boxes and random false positives,
clipped to the image as torchvision's postprocess does), so the writer, parser and evaluator can be checked
without a GPU. The difference then measures only the CSV format and its output precision; clipping of
out-of-image boxes is covered by `tests/test_submission.py`.
The CSV goes to the git-ignored `submissions/` folder; the comparison to
`results/ch-05-submission/roundtrip_<name>.json`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from movd import data, submission  # noqa: E402
from movd.evaluate import evaluate_split, load_predictions  # noqa: E402
from movd.tv_train import save_json  # noqa: E402

TOLERANCE = 1e-6


def detector_box(x, y, w, h, width, height):
    """xywh -> xywh clipped to the image, as torchvision's `clip_boxes_to_image` does. None if empty."""
    x0, y0 = min(max(x, 0.0), width), min(max(y, 0.0), height)
    x1, y1 = min(max(x + w, 0.0), width), min(max(y + h, 0.0), height)
    if x1 <= x0 or y1 <= y0:
        return None
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def synthetic_predictions(
    coco: dict, image_ids: list[int], seed: int = data.SEED
) -> list[dict]:
    """Jittered GT boxes (80 % of targets, 10 % with a wrong class) plus 30 random false positives per
    frame, clipped to the image like real detector output. Scores are random, so ties are rare."""
    rng = np.random.default_rng(seed)
    images = {im["id"]: im for im in coco["images"]}
    wanted = set(image_ids)
    raw = []
    for ann in coco["annotations"]:
        if ann["image_id"] not in wanted or ann.get("iscrowd", 0) or rng.random() > 0.8:
            continue
        x, y, w, h = ann["bbox"]
        jx, jy, jw, jh = rng.normal(0.0, 0.08, 4) * [w, h, w, h]
        wrong_class = rng.random() < 0.1
        raw.append(
            (
                ann["image_id"],
                int(rng.integers(1, 9)) if wrong_class else ann["category_id"],
                (x + jx, y + jy, max(w + jw, 1.0), max(h + jh, 1.0)),
                float(rng.uniform(0.2, 1.0)),
            )
        )
    for image_id in image_ids:
        width, height = images[image_id]["width"], images[image_id]["height"]
        for _ in range(30):
            w, h = rng.uniform(4, 120, 2)
            x, y = rng.uniform(-40, width), rng.uniform(-40, height)
            raw.append(
                (
                    image_id,
                    int(rng.integers(1, 9)),
                    (x, y, w, h),
                    float(rng.uniform(0.0, 0.6)),
                )
            )
    predictions = []
    for image_id, category_id, box, score in raw:
        image = images[image_id]
        bbox = detector_box(*box, image["width"], image["height"])
        if bbox is not None:
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": score,
                }
            )
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path)
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    args = parser.parse_args()

    coco = data.load_coco("val", args.dataset_root)
    image_ids = data.split_image_ids("val_full")
    direct = (
        synthetic_predictions(coco, image_ids)
        if args.synthetic
        else load_predictions(args.predictions)
    )
    images = submission.coco_image_table(coco, image_ids)

    rows, stats = submission.build_rows(direct, images)
    csv_path = Path("submissions") / f"val_roundtrip_{args.name}.csv"
    submission.write_csv(csv_path, rows)
    table = submission.read_csv(csv_path)
    problems, summary = submission.validate_rows(table, images)
    parsed = submission.rows_to_predictions(table[1:], images)

    direct_result = evaluate_split(direct, "val_full", args.dataset_root)
    parsed_result = evaluate_split(parsed, "val_full", args.dataset_root)
    diff = {
        k: abs(direct_result["metrics"][k] - parsed_result["metrics"][k])
        for k in direct_result["metrics"]
        if direct_result["metrics"][k] is not None
    }
    passed = not problems and max(diff.values()) <= TOLERANCE
    out = Path("results") / "ch-05-submission" / f"roundtrip_{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        out,
        {
            "source": "synthetic (seed 42)"
            if args.synthetic
            else args.predictions.as_posix(),
            "split": "val_full",
            "tolerance": TOLERANCE,
            "passed": passed,
            "max_abs_difference": max(diff.values()),
            "abs_difference": diff,
            "direct": direct_result["metrics"],
            "roundtrip": parsed_result["metrics"],
            "writer": stats,
            "validator": {"valid": not problems, "problems": problems, **summary},
        },
    )
    for k in direct_result["metrics"]:
        print(
            f"{k:<6} direct {direct_result['metrics'][k]!s:<22} round-trip {parsed_result['metrics'][k]!s:<22}"
        )
    print(f"writer: {stats}")
    print(
        f"max |difference| = {max(diff.values()):.3g} (tolerance {TOLERANCE}); validator problems: {problems}"
    )
    print(f"[{'OK' if passed else 'FAIL'}] wrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
