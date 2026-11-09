"""Second, independent check of a Kaggle submission CSV (CH-14). Standard library only; never imports `movd`.

    python scripts/check_submission_independent.py <csv> --predictions <test_predictions.json> \
        --run-json <run.json> --expected-checkpoint-sha256 <sha256> [--out <summary.json>]

It re-implements the format rules from the Kaggle Data text / plan.md §4 instead of calling
`movd.submission`:

* header exactly `image_id,PredictionString`; exactly the 3,811 `test_images.csv` ids, unique, in that order;
* groups of 6 tokens `class score x y width height`; class an integer 0-7; finite score in [0, 1];
  finite box with width, height > 0 inside the original image (`test_images.csv` width/height, 1e-6 px);
  at most 500 groups per row;
* lossless numbers: every float token is the shortest round-trip form (`repr(float(t)) == t`), every class
  token a plain integer;
* with `--predictions`: each image's groups are exactly the top-500 (by score) of its COCO-results
  predictions (category_id - 1, boxes clipped to the frame, empty boxes dropped), compared value by value;
* with `--run-json` / `--expected-checkpoint-sha256`: the recorded checkpoint hash, predictions hash and CSV
  hash match.

Exit code 0 = every check passed, 1 = at least one failed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "multiple-object-video-detection-hw-2-ucu-ai-2026"
HEADER = ["image_id", "PredictionString"]
MAX_DETS = 500
NUM_CLASSES = 8
TOL = 1e-6


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clip_box(x, y, w, h, width, height):
    """Clip [x, y, w, h] to the frame; keep w/h exact when that side needed no clipping."""
    x0, x1 = min(max(x, 0.0), width), min(max(x + w, 0.0), width)
    y0, y1 = min(max(y, 0.0), height), min(max(y + h, 0.0), height)
    w = w if (x0, x1) == (x, x + w) else x1 - x0
    h = h if (y0, y1) == (y, y + h) else y1 - y0
    return x0, y0, w, h


def expected_groups(predictions, sizes):
    """image key -> list of (class, score, x, y, w, h) the CSV must hold, from COCO-results predictions."""
    by_image = defaultdict(list)
    for p in predictions:
        key, (width, height) = sizes["by_id"][p["image_id"]]
        x, y, w, h = (float(v) for v in p["bbox"])
        score = float(p["score"])
        if not (math.isfinite(score) and 0.0 <= score <= 1.0):
            continue
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w < 0 or h < 0:
            continue
        x, y, w, h = clip_box(x, y, w, h, width, height)
        if w <= 0 or h <= 0:
            continue
        by_image[key].append((int(p["category_id"]) - 1, score, x, y, w, h))
    return {
        k: sorted(v, key=lambda d: (-d[1], d[0], *d[2:]))[:MAX_DETS]
        for k, v in by_image.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", type=Path)
    parser.add_argument("--test-images", type=Path, default=DATASET / "test_images.csv")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--run-json", type=Path, default=None)
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    failures: list[str] = []

    def fail(message: str) -> None:
        if len(failures) < 50:
            failures.append(message)
        else:
            failures[-1] = "... more failures"

    with open(args.test_images, newline="", encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    keys = [r["image_id"] for r in table]
    sizes = {
        "by_key": {r["image_id"]: (int(r["width"]), int(r["height"])) for r in table},
        "by_id": {
            i + 1: (r["image_id"], (int(r["width"]), int(r["height"])))
            for i, r in enumerate(table)
        },
    }

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != HEADER:
        fail(f"header {rows[0] if rows else None!r} != {HEADER}")
    body = rows[1:]
    ids = [r[0] if r else "" for r in body]
    if len(body) != len(keys):
        fail(f"{len(body)} rows, expected {len(keys)}")
    if len(set(ids)) != len(ids):
        fail(f"{len(ids) - len(set(ids))} duplicate ids")
    if set(ids) - set(keys):
        fail(f"{len(set(ids) - set(keys))} unknown ids")
    if set(keys) - set(ids):
        fail(f"{len(set(keys) - set(ids))} missing ids")
    if ids != keys:
        fail("row order differs from test_images.csv")

    parsed: dict[str, list[tuple]] = {}
    groups_total, max_groups, empty_rows, classes_seen = 0, 0, 0, set()
    for n, row in enumerate(body, start=2):
        if len(row) != 2:
            fail(f"line {n}: {len(row)} columns")
            continue
        key, text = row
        tokens = text.split(" ") if text else []
        if text and (text != text.strip() or "" in tokens):
            fail(f"line {n}: extra whitespace")
        if len(tokens) % 6:
            fail(f"line {n}: {len(tokens)} tokens, not a multiple of 6")
            continue
        count = len(tokens) // 6
        groups_total += count
        max_groups = max(max_groups, count)
        empty_rows += count == 0
        if count > MAX_DETS:
            fail(f"line {n}: {count} groups > {MAX_DETS}")
        width, height = sizes["by_key"].get(key, (None, None))
        dets = []
        for g in range(count):
            c, *nums = tokens[6 * g : 6 * g + 6]
            if not c.isdigit() or str(int(c)) != c or not 0 <= int(c) < NUM_CLASSES:
                fail(f"line {n} group {g}: class {c!r}")
                continue
            try:
                values = [float(t) for t in nums]
            except ValueError:
                fail(f"line {n} group {g}: non-numeric {nums}")
                continue
            for t, v in zip(nums, values):
                if not math.isfinite(v):
                    fail(f"line {n} group {g}: non-finite {t}")
                elif repr(v) != t:
                    fail(
                        f"line {n} group {g}: {t!r} is not the lossless form {repr(v)!r}"
                    )
            score, x, y, w, h = values
            if not 0.0 <= score <= 1.0:
                fail(f"line {n} group {g}: score {score}")
            if not (w > 0 and h > 0):
                fail(f"line {n} group {g}: non-positive box {w} x {h}")
            if width is not None and not (
                x >= -TOL
                and y >= -TOL
                and x + w <= width + TOL
                and y + h <= height + TOL
            ):
                fail(f"line {n} group {g}: box {x, y, w, h} outside {width}x{height}")
            classes_seen.add(int(c))
            dets.append((int(c), score, x, y, w, h))
        parsed[key] = dets

    report = {
        "csv": args.csv.resolve().as_posix(),
        "csv_bytes": args.csv.stat().st_size,
        "csv_sha256": file_sha256(args.csv),
        "rows": len(body),
        "groups": groups_total,
        "max_groups_per_row": max_groups,
        "empty_rows": empty_rows,
        "classes_seen": sorted(classes_seen),
    }

    if args.predictions:
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
        report["predictions"] = args.predictions.resolve().as_posix()
        report["predictions_sha256"] = file_sha256(args.predictions)
        report["predictions_count"] = len(predictions)
        expected = expected_groups(predictions, sizes)
        mismatched = [k for k in keys if parsed.get(k, []) != expected.get(k, [])]
        report["rows_differing_from_predictions"] = len(mismatched)
        if mismatched:
            fail(
                f"{len(mismatched)} rows differ from the predictions, first {mismatched[0]}"
            )
    if args.run_json:
        run = json.loads(args.run_json.read_text(encoding="utf-8"))
        report["run_json"] = args.run_json.resolve().as_posix()
        report["run_checkpoint_sha256"] = run.get("checkpoint_sha256")
        for field, actual in (
            ("csv_sha256", report["csv_sha256"]),
            ("test_predictions_sha256", report.get("predictions_sha256")),
        ):
            if field in run and actual is not None and run[field] != actual:
                fail(f"run.json {field} {run[field]} != {actual}")
    if args.expected_checkpoint_sha256:
        recorded = report.get("run_checkpoint_sha256")
        if recorded != args.expected_checkpoint_sha256:
            fail(
                f"checkpoint sha256 {recorded} != expected {args.expected_checkpoint_sha256}"
            )

    report["failures"] = failures
    report["valid"] = not failures
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
