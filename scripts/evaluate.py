"""Evaluate a COCO-results JSON on a frozen val split with the project evaluator (movd/evaluate.py).

Example:
    python scripts/evaluate.py outputs/<run_id>/val_full_predictions.json --split val_full --out results/<run_id>/eval

Writes <out>/metrics.json and <out>/per_class.csv and prints a summary. Headline: maxDets 500.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import evaluate as ev


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=(
            "Predictions: a JSON list of {image_id, category_id (1-8), bbox [x, y, w, h] in original pixels, score}, "
            "with COCO image ids from val/annotations_coco.json. Metrics: mAP (AP@0.50:0.05:0.95), AP50, AP75, "
            f"AP_small/medium/large, per-class AP and density bins, at maxDets {ev.HEADLINE_MAX_DETS} (headline) "
            f"and {ev.LEGACY_MAX_DETS} (legacy comparison only). Crowd/ignore regions follow the CH-02 policy."
        ),
    )
    parser.add_argument("predictions", type=Path, help="COCO-results JSON file")
    parser.add_argument(
        "--split",
        default="val_full",
        choices=["val_full", "val_fast"],
        help="frozen frame split (default: val_full, used for every headline number)",
    )
    parser.add_argument(
        "--out", type=Path, help="output folder for metrics.json and per_class.csv"
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="dataset root (default: ./multiple-object-video-detection-hw-2-ucu-ai-2026)",
    )
    args = parser.parse_args(argv)

    result = ev.evaluate_split(
        ev.load_predictions(args.predictions), args.split, args.root
    )
    result["predictions_file"] = args.predictions.as_posix()
    print(
        f"{args.split}: {result['num_images']} images, {result['num_predictions']} predictions "
        f"({result['num_predictions_outside_split']} outside the split dropped)"
    )
    print(ev.format_summary(result))
    if args.out:
        ev.write_outputs(result, args.out)
        print(f"wrote {args.out / 'metrics.json'} and {args.out / 'per_class.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
