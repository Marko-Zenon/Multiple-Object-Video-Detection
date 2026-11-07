"""Generate (or validate) a canonical local sample_submission.csv from test_images.csv.

    python scripts/make_sample_submission.py           # write <dataset root>/sample_submission.csv, then validate it
    python scripts/make_sample_submission.py --check   # validate the existing file only

Why: the official Kaggle Data description lists sample_submission.csv, but the competition's "Download All"
archive does not contain it (77,434 entries, checked in CH-02). This file is rebuilt from the documented
schema: columns `image_id,PredictionString`, one row per test image in `test_images.csv` order, and an
empty PredictionString (which is valid). It lives in the git-ignored dataset folder and is never committed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import data  # noqa: E402

HEADER = ["image_id", "PredictionString"]


def expected_ids(root: Path | None) -> list[str]:
    return [row["image_key"] for row in data.load_test_images(root)]


def write(path: Path, ids: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows([image_id, ""] for image_id in ids)


def validate(path: Path, ids: list[str]) -> list[str]:
    """Return a list of problems (empty = valid)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return ["file is empty"]
    problems = []
    if rows[0] != HEADER:
        problems.append(f"header is {rows[0]}, expected {HEADER}")
    body = rows[1:]
    got = [r[0] if r else "" for r in body]
    if len(body) != len(ids):
        problems.append(f"{len(body)} data rows, expected {len(ids)}")
    if len(set(got)) != len(got):
        problems.append("duplicate image_id values")
    if missing := set(ids) - set(got):
        problems.append(f"{len(missing)} missing image_id values")
    if extra := set(got) - set(ids):
        problems.append(f"{len(extra)} unknown image_id values")
    if got != ids:
        problems.append("image_id order differs from test_images.csv")
    if any(len(r) != 2 for r in body):
        problems.append("some rows do not have exactly 2 columns")
    if any(len(r) == 2 and r[1] != "" for r in body):
        problems.append("some PredictionString values are not empty")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=None, help="dataset root (default: repo dataset folder)")
    parser.add_argument("--check", action="store_true", help="only validate the existing file")
    args = parser.parse_args()

    root = args.root if args.root is not None else data.DATASET_ROOT
    path = root / "sample_submission.csv"
    ids = expected_ids(root)
    if not args.check:
        write(path, ids)
        print(f"wrote {path}")
    if not path.exists():
        print(f"[FAIL] {path} does not exist")
        return 1
    problems = validate(path, ids)
    for problem in problems:
        print(f"[FAIL] {problem}")
    if not problems:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"[OK] {path.name}: header {','.join(HEADER)}, {len(ids)} rows, ids unique and in "
              f"test_images.csv order, all PredictionString empty, sha256 {digest}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
