"""Check that the VisDrone-VID homework dataset is extracted where the code expects it (M-DATA check).

Standard library only, so it runs before any environment is installed:

    python scripts/check_dataset.py            # default root next to this repo
    python scripts/check_dataset.py --root D:/data/multiple-object-video-detection-hw-2-ucu-ai-2026
    python scripts/check_dataset.py --skip-json  # skip loading the ~280 MB train COCO JSON

Exit code 0 means every check passed; 1 means at least one failed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "multiple-object-video-detection-hw-2-ucu-ai-2026"

# Expected counts, taken from the provided ZIP (see plan.md §2).
EXPECTED = {
    "train": {"images": 24_201, "clips": 56, "ignore_clips": 50, "annotations": 1_616_260},
    "val": {"images": 2_846, "clips": 7, "ignore_clips": 5, "annotations": 146_092},
    "test": {"images": 3_811, "clips": 48},
}
CLASS_NAMES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]


class Checker:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, ok: bool, message: str) -> bool:
        print(f"[{'OK' if ok else 'FAIL'}] {message}")
        if not ok:
            self.failures += 1
        return ok


def clip_dirs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []


def count_files(folder: Path, suffix: str) -> int:
    return sum(1 for _ in folder.rglob(f"*{suffix}")) if folder.is_dir() else 0


def check_split(c: Checker, root: Path, split: str, skip_json: bool) -> None:
    exp = EXPECTED[split]
    split_dir = root / split
    if not c.check(split_dir.is_dir(), f"{split}/ exists"):
        return

    images_dir = split_dir / "images"
    clips = clip_dirs(images_dir)
    n_images = count_files(images_dir, ".jpg")
    c.check(len(clips) == exp["clips"], f"{split}: {len(clips)} clips (expected {exp['clips']})")
    c.check(n_images == exp["images"], f"{split}: {n_images} images (expected {exp['images']})")

    if split == "test":
        return

    labels_dir = split_dir / "labels"
    n_labels = count_files(labels_dir, ".txt")
    c.check(n_labels == exp["images"], f"{split}: {n_labels} YOLO label files (expected {exp['images']})")

    ignore_clips = [p for p in clip_dirs(split_dir / "ignore") if any(p.iterdir())]
    c.check(
        len(ignore_clips) == exp["ignore_clips"],
        f"{split}: ignore/ files for {len(ignore_clips)} clips (expected {exp['ignore_clips']})",
    )
    c.check((split_dir / "data.yaml").is_file(), f"{split}/data.yaml exists")

    ann_path = split_dir / "annotations_coco.json"
    if not c.check(ann_path.is_file(), f"{split}/annotations_coco.json exists"):
        return
    if skip_json:
        print(f"[SKIP] {split}: COCO JSON not loaded (--skip-json)")
        return
    try:
        with ann_path.open(encoding="utf-8") as f:
            coco = json.load(f)
    except (OSError, json.JSONDecodeError) as err:
        c.check(False, f"{split}: COCO JSON loads ({err})")
        return
    n_img, n_ann = len(coco.get("images", [])), len(coco.get("annotations", []))
    names = [cat["name"] for cat in sorted(coco.get("categories", []), key=lambda cat: cat["id"])]
    c.check(n_img == exp["images"], f"{split}: COCO JSON has {n_img} images (expected {exp['images']})")
    c.check(n_ann == exp["annotations"], f"{split}: COCO JSON has {n_ann} annotations (expected {exp['annotations']})")
    c.check(names == CLASS_NAMES, f"{split}: COCO categories {names}")


def check_test_csv(c: Checker, root: Path) -> None:
    csv_path = root / "test_images.csv"
    if not c.check(csv_path.is_file(), "test_images.csv exists"):
        return
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected = EXPECTED["test"]["images"]
    c.check(len(rows) == expected, f"test_images.csv: {len(rows)} rows (expected {expected})")
    missing = [r["file_name"] for r in rows if not (root / r["file_name"]).is_file()]
    c.check(not missing, f"test_images.csv: every file_name exists ({len(missing)} missing)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / DATASET_NAME, help="dataset root directory")
    parser.add_argument("--skip-json", action="store_true", help="do not load the COCO JSON files")
    args = parser.parse_args()
    root: Path = args.root

    c = Checker()
    print(f"Dataset root: {root}")
    if not c.check(root.is_dir(), "dataset root exists"):
        return 1
    c.check(not (root / DATASET_NAME).exists(), "no double nesting (root/<dataset name>/ absent)")
    for stray in ("train", "val", "test", "test_images.csv"):
        c.check(not (REPO_ROOT / stray).exists(), f"no stray {stray} in the repo root")

    for split in ("train", "val", "test"):
        check_split(c, root, split, args.skip_json)
    check_test_csv(c, root)

    print(f"\n{'PASSED' if c.failures == 0 else f'FAILED ({c.failures} checks)'}")
    return 0 if c.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
