"""Kaggle submission writer, parser and validator (CH-05).

Format (frozen by CH-02, gate G3, `plan.md` §4 / `configs/protocol.json` "submission"):

* header `image_id,PredictionString`, exactly one row per `test_images.csv` image, in that file's order;
* `PredictionString` = space-separated groups `class score x y width height`, where class is 0–7
  (`category_id - 1`) and `(x, y)` is the top-left corner in original-image pixels;
* an empty `PredictionString` is valid; Kaggle scores at most 500 detections per image.

Input is the project's COCO-results interchange format (`plan.md` §4):
`[{"image_id": int, "category_id": 1-8, "bbox": [x, y, w, h] original pixels, "score": float}]`.
For test predictions `image_id` is the 1-based row number of `test_images.csv`
(`movd.data.load_test_images()[i]["id"]`); the row's string `image_key` goes into the CSV.

Writer rules, in order, per detection:
1. reject a detection with a non-finite score or a score outside [0, 1];
2. reject a box with a non-finite coordinate or a negative width or height;
3. clip the box to the image (`0 <= x0 < x1 <= width`, same for y);
4. reject boxes whose width or height is not positive after clipping (zero area);
5. sort the image's detections by full-precision score, descending; ties are broken by class and box so the
   order never depends on input order;
6. keep the first `max_dets` (500) over all classes.

Output precision: lossless. Every float is written as Python `repr`, which parses back to the identical
value, so the val round-trip equals direct evaluation exactly. Rounding is not safe: with the real CH-07
frcnn_rpn_pre3000 val_full predictions some COCO matches sit exactly on an IoU threshold, and rounding
boxes to 5, 8 or even 12 decimals changed AP50 by 4.6e-6, 4.5e-6 and 1.5e-6 (tolerance 1e-6); 17 digits
gave 0. The cost is about 90 characters per detection instead of 52. Clipped boxes stay inside the image,
so the validator's bounds tolerance is only for float noise in `x + width` (`BOUNDS_TOLERANCE`).

Rejected detections are counted, never silently written. The validator (`validate_rows`) uses its own
parsing and does not call the writer, so it can check any CSV, including ones not produced here.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path

from movd import data

HEADER = ["image_id", "PredictionString"]
MAX_DETS = 500
FLOAT_FORMAT = "repr (lossless)"
# The writer never exceeds the image bounds; this only absorbs float noise in `x + width`.
BOUNDS_TOLERANCE = 1e-6
GROUP_SIZE = 6


# ---------------------------------------------------------------- image tables


def test_image_table(root: str | Path | None = None) -> list[dict]:
    """`[{"id", "key", "width", "height"}]` for the test set, in `test_images.csv` order."""
    return [
        {
            "id": r["id"],
            "key": r["image_key"],
            "width": r["width"],
            "height": r["height"],
        }
        for r in data.load_test_images(root)
    ]


def coco_image_table(coco: dict, image_ids: Iterable[int] | None = None) -> list[dict]:
    """Same table for a COCO split (val round-trip). The key is the COCO image id as a string."""
    images = {im["id"]: im for im in coco["images"]}
    ids = sorted(images) if image_ids is None else list(image_ids)
    return [
        {
            "id": i,
            "key": str(i),
            "width": images[i]["width"],
            "height": images[i]["height"],
        }
        for i in ids
    ]


# ---------------------------------------------------------------- writer


def clean_detection(pred: dict, width: int, height: int) -> tuple[str, tuple | None]:
    """Apply writer rules 1–4 to one COCO-results detection.

    Returns `(reason, None)` for a rejected detection, or `("ok", (score, class, x, y, w, h))` with the
    full-precision score (for sorting) and the rounded box. A category id outside 1–8 raises, because
    it means a label-map shift, not a bad detection.
    """
    class_id = data.category_to_class(pred["category_id"])
    score = float(pred["score"])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return "bad_score", None
    x, y, w, h = (float(v) for v in pred["bbox"])
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w < 0 or h < 0:
        return "bad_box", None
    x0, y0 = min(max(x, 0.0), width), min(max(y, 0.0), height)
    x1, y1 = min(max(x + w, 0.0), width), min(max(y + h, 0.0), height)
    # A box that needed no clipping keeps its exact width/height (x1 - x0 can differ by one ulp).
    if (x0, x1) != (x, x + w):
        w = x1 - x0
    if (y0, y1) != (y, y + h):
        h = y1 - y0
    if w <= 0 or h <= 0:
        return "empty_box", None
    return "ok", (score, class_id, x0, y0, w, h)


def format_group(det: tuple) -> str:
    """`class score x y w h`; floats as `repr`, the shortest string that parses back to the same value."""
    score, class_id, x, y, w, h = det
    return " ".join([str(class_id), *(repr(float(v)) for v in (score, x, y, w, h))])


def build_rows(
    predictions: Sequence[dict], images: Sequence[dict], max_dets: int = MAX_DETS
) -> tuple[list[list[str]], dict]:
    """COCO-results predictions -> CSV rows `[key, PredictionString]` in `images` order, plus counters.

    Raises on a prediction for an unknown image id or with a category id outside 1–8 (a label-map bug
    must not pass silently). Images without valid detections get an empty string.
    """
    by_id = {im["id"]: im for im in images}
    if len(by_id) != len(images):
        raise ValueError("duplicate image ids in the image table")
    keys = [im["key"] for im in images]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate image keys in the image table")

    kept = defaultdict(list)
    stats = Counter()
    for i, pred in enumerate(predictions):
        image = by_id.get(pred["image_id"])
        if image is None:
            raise ValueError(
                f"prediction {i}: image_id {pred['image_id']!r} is not in the image table"
            )
        reason, det = clean_detection(pred, image["width"], image["height"])
        stats[reason] += 1
        if det is not None:
            kept[image["id"]].append(det)

    rows = []
    for image in images:
        dets = sorted(kept.get(image["id"], []), key=lambda d: (-d[0], *d[1:]))
        stats["over_cap"] += max(0, len(dets) - max_dets)
        dets = dets[:max_dets]
        stats["written"] += len(dets)
        stats["empty_rows"] += not dets
        rows.append([image["key"], " ".join(format_group(d) for d in dets)])
    return rows, {
        "input_predictions": len(predictions),
        "accepted": stats["ok"],
        "rejected_bad_score": stats["bad_score"],
        "rejected_bad_box": stats["bad_box"],
        "rejected_empty_box": stats["empty_box"],
        "dropped_over_cap": stats["over_cap"],
        "written": stats["written"],
        "rows": len(rows),
        "empty_rows": stats["empty_rows"],
        "max_dets": max_dets,
        "float_format": FLOAT_FORMAT,
    }


def write_csv(path: str | Path, rows: Sequence[Sequence[str]]) -> None:
    """Write the CSV atomically (temp file, then rename) with `\\n` line endings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
    temp.replace(path)


# ---------------------------------------------------------------- parser


def parse_prediction_string(
    text: str,
) -> list[tuple[int, float, float, float, float, float]]:
    """`"c s x y w h c s x y w h ..."` -> `[(class, score, x, y, w, h), ...]`. Empty string -> `[]`.

    Raises `ValueError` on a token count that is not a multiple of 6, a non-integer class or a
    non-numeric value. Range checks are the validator's job.
    """
    tokens = text.split()
    if len(tokens) % GROUP_SIZE:
        raise ValueError(f"{len(tokens)} tokens is not a multiple of {GROUP_SIZE}")
    groups = []
    for start in range(0, len(tokens), GROUP_SIZE):
        cls, *values = tokens[start : start + GROUP_SIZE]
        if not cls.isdigit():
            raise ValueError(
                f"group {start // GROUP_SIZE}: class {cls!r} is not a non-negative integer"
            )
        groups.append((int(cls), *(float(v) for v in values)))
    return groups


def read_csv(path: str | Path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def rows_to_predictions(
    rows: Sequence[Sequence[str]], images: Sequence[dict]
) -> list[dict]:
    """Parse submission data rows (no header) back into COCO-results predictions (category = class + 1)."""
    id_by_key = {im["key"]: im["id"] for im in images}
    predictions = []
    for key, text in rows:
        for cls, score, x, y, w, h in parse_prediction_string(text):
            predictions.append(
                {
                    "image_id": id_by_key[key],
                    "category_id": data.class_to_category(cls),
                    "bbox": [x, y, w, h],
                    "score": score,
                }
            )
    return predictions


# ---------------------------------------------------------------- validator


def validate_rows(
    rows: Sequence[Sequence[str]],
    images: Sequence[dict],
    max_dets: int = MAX_DETS,
    tolerance: float = BOUNDS_TOLERANCE,
    max_listed: int = 5,
) -> tuple[list[str], dict]:
    """Validate a full CSV (header included). Returns `(problems, summary)`; no problems = valid.

    Checks: header; row count; duplicate, missing and unknown ids; row order; 2 columns per row;
    group arity; integer class 0–7; finite score in [0, 1]; finite box with w, h > 0 inside the image
    (`tolerance` px); at most `max_dets` groups. Score order is reported in the summary, not as a problem,
    because Kaggle does not require it.
    """
    problems: list[str] = []
    if not rows:
        return ["file is empty"], {}
    if list(rows[0]) != HEADER:
        problems.append(f"header is {list(rows[0])}, expected {HEADER}")
    body = rows[1:]
    expected = [im["key"] for im in images]
    by_key = {im["key"]: im for im in images}
    got = [r[0] if r else "" for r in body]

    if len(body) != len(expected):
        problems.append(f"{len(body)} data rows, expected {len(expected)}")
    duplicates = [k for k, n in Counter(got).items() if n > 1]
    if duplicates:
        problems.append(
            f"{len(duplicates)} duplicate image_id values, e.g. {duplicates[:max_listed]}"
        )
    missing = sorted(set(expected) - set(got))
    if missing:
        problems.append(
            f"{len(missing)} missing image_id values, e.g. {missing[:max_listed]}"
        )
    unknown = sorted(set(got) - set(expected))
    if unknown:
        problems.append(
            f"{len(unknown)} unknown image_id values, e.g. {unknown[:max_listed]}"
        )
    if not (duplicates or missing or unknown) and got != expected:
        problems.append(
            "image_id order differs from the expected (test_images.csv) order"
        )

    row_problems: list[str] = []
    groups_total, empty_rows, max_groups, unsorted_rows = 0, 0, 0, 0
    class_counts = Counter()
    for line, row in enumerate(body, 2):
        if len(row) != 2:
            row_problems.append(f"line {line}: {len(row)} columns, expected 2")
            continue
        key, text = row
        image = by_key.get(key)
        tokens = text.split()
        if len(tokens) % GROUP_SIZE:
            row_problems.append(
                f"line {line}: {len(tokens)} tokens, not a multiple of {GROUP_SIZE}"
            )
            continue
        n = len(tokens) // GROUP_SIZE
        groups_total += n
        empty_rows += n == 0
        max_groups = max(max_groups, n)
        if n > max_dets:
            row_problems.append(f"line {line}: {n} detections, more than {max_dets}")
        scores = []
        for g in range(n):
            cls, *values = tokens[g * GROUP_SIZE : (g + 1) * GROUP_SIZE]
            where = f"line {line} group {g}"
            if not cls.isdigit() or not 0 <= int(cls) < data.NUM_CLASSES:
                row_problems.append(
                    f"{where}: class {cls!r} is not an integer 0-{data.NUM_CLASSES - 1}"
                )
                continue
            try:
                score, x, y, w, h = (float(v) for v in values)
            except ValueError:
                row_problems.append(f"{where}: non-numeric value in {values}")
                continue
            class_counts[int(cls)] += 1
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                row_problems.append(
                    f"{where}: score {values[0]} not a finite value in [0, 1]"
                )
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                row_problems.append(f"{where}: non-finite box {values[1:]}")
                continue
            if w <= 0 or h <= 0:
                row_problems.append(f"{where}: width/height must be > 0, got {w} {h}")
            if image is not None and (
                x < -tolerance
                or y < -tolerance
                or x + w > image["width"] + tolerance
                or y + h > image["height"] + tolerance
            ):
                row_problems.append(
                    f"{where}: box {values[1:]} outside the {image['width']}x{image['height']} image"
                )
            scores.append(score)
        unsorted_rows += any(a < b for a, b in pairwise(scores))
    if row_problems:
        problems.append(
            f"{len(row_problems)} row/group problems, e.g.: "
            + "; ".join(row_problems[:max_listed])
        )

    summary = {
        "data_rows": len(body),
        "expected_rows": len(expected),
        "detections": groups_total,
        "empty_rows": empty_rows,
        "max_detections_in_a_row": max_groups,
        "rows_not_sorted_by_score": unsorted_rows,
        "detections_per_class": {
            data.CLASS_NAMES[c]: class_counts[c] for c in range(data.NUM_CLASSES)
        },
    }
    return problems, summary
