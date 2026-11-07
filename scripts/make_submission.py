"""COCO-results test predictions -> Kaggle submission CSV, then validate it (CH-05).

    python scripts/make_submission.py outputs/ch-05-submission/<name>/test_predictions.json
    python scripts/make_submission.py <predictions.json> --out submissions/<file>.csv

The CSV goes to the git-ignored `submissions/` folder (default `submissions/<predictions folder name>.csv`).
Writer rules (clipping, rejection, ordering, the 500 cap, precision) are in `movd/submission.py`.
A summary with the file's sha256 and the writer and validator counters is written to
`results/ch-05-submission/<csv stem>/submission.json`. Exit code 0 only if the written file validates.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import submission  # noqa: E402
from movd.evaluate import load_predictions  # noqa: E402
from movd.tv_train import save_json  # noqa: E402

RESULTS_ROOT = Path("results") / "ch-05-submission"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "predictions",
        type=Path,
        help="COCO-results JSON, image_id = test_images.csv row",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--max-dets", type=int, default=submission.MAX_DETS)
    args = parser.parse_args()

    out = args.out or Path("submissions") / f"{args.predictions.parent.name}.csv"
    if out.suffix != ".csv":
        raise SystemExit("--out must be a .csv file")
    images = submission.test_image_table(args.dataset_root)
    rows, stats = submission.build_rows(
        load_predictions(args.predictions), images, args.max_dets
    )
    submission.write_csv(out, rows)

    problems, summary = submission.validate_rows(
        submission.read_csv(out), images, args.max_dets
    )
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    results = RESULTS_ROOT / out.stem
    results.mkdir(parents=True, exist_ok=True)
    save_json(
        results / "submission.json",
        {
            "predictions": args.predictions.as_posix(),
            "submission": out.as_posix(),
            "sha256": digest,
            "bytes": out.stat().st_size,
            "writer": stats,
            "validator": {"valid": not problems, "problems": problems, **summary},
        },
    )
    print(
        f"wrote {out}: {stats['rows']} rows, {stats['written']} detections, {stats['empty_rows']} empty rows"
    )
    print(
        f"rejected: score {stats['rejected_bad_score']}, box {stats['rejected_bad_box']}, "
        f"zero-area {stats['rejected_empty_box']}; over the {args.max_dets} cap: {stats['dropped_over_cap']}"
    )
    for problem in problems:
        print(f"[FAIL] {problem}")
    if not problems:
        print(f"[OK] valid, sha256 {digest}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
