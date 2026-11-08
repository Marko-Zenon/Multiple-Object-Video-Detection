"""Validate a Kaggle submission CSV against test_images.csv (CH-05). Independent of the inference code.

    python scripts/validate_submission.py submissions/<file>.csv
    python scripts/validate_submission.py submissions/<file>.csv --dataset-root <dataset folder>

Checks (see `movd.submission.validate_rows`): header `image_id,PredictionString`; exactly one row per
test image, no duplicate, missing or unknown ids, in `test_images.csv` order; 2 columns per row; groups of
6 values; integer class 0–7; finite score in [0, 1]; finite box with width, height > 0 inside the original
image (tolerance 1e-6 px); at most 500 groups per row. Exit code 0 = valid, 1 = invalid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import submission  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--max-dets", type=int, default=submission.MAX_DETS)
    args = parser.parse_args()

    images = submission.test_image_table(args.dataset_root)
    problems, summary = submission.validate_rows(
        submission.read_csv(args.csv), images, args.max_dets
    )
    for problem in problems:
        print(f"[FAIL] {problem}")
    print(json.dumps(summary, indent=2))
    if not problems:
        digest = hashlib.sha256(args.csv.read_bytes()).hexdigest()
        print(
            f"[OK] {args.csv}: {summary['data_rows']} rows, {summary['detections']} detections, sha256 {digest}"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
