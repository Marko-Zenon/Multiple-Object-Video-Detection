"""Shared data layer: paths, COCO and ignore loaders, label map, frozen splits, ignore-region policy.

Frozen by CH-02. The protocol values live in `configs/protocol.json` and `plan.md` §4; this module
implements them. Later changes go through a PR reviewed by the other student (plan.md §8).

Verified dataset facts (CH-02 inspection, evidence in `results/ch-02-protocol/`):

* COCO `category_id` 1–8 = person, bicycle, car, van, truck, tricycle, bus, motor. They are merged
  VisDrone classes (`orig_category_id`): 1 pedestrian + 2 people → person, 3 → bicycle, 4 → car,
  5 → van, 6 → truck, 7 tricycle + 8 awning-tricycle → tricycle, 9 → bus, 10 → motor.
  Submission / YOLO class = `category_id - 1` (0–7). The YOLO `labels/` files already use it.
* Ignore regions are VisDrone "ignored regions" (orig 0) and "others" (orig 11). They exist twice:
  - `<split>/ignore/<clip>/<frame>.txt`: one line per region, `cx cy w h` normalised to [0, 1] (no class);
  - `annotations_coco.json`: each region repeated 8 times (once per category) with `iscrowd=1, ignore=1`.
  Both sources hold exactly the same regions (verified on all val frames).

Ignore-region policy (frozen):

* Training: crowd annotations are never used as targets. Each ignore region is filled with the ImageNet
  mean colour, except pixels inside a labelled target box, which stay untouched. Unlabelled objects in
  the region therefore cannot be learned as background, and no labelled target is corrupted.
  See `mask_ignore_regions` and `training_sample`.
* Evaluation: the COCO GT JSON is used unchanged, crowd annotations included. pycocotools then ignores
  any detection matched to a crowd region (overlap = intersection / detection area), for every class,
  because each region is replicated per category. Implemented by CH-03 in `movd/evaluate.py`.
  This matches the official Kaggle Data description: detections inside ignored regions are not penalized,
  ignored COCO annotations use `iscrowd=1`, and at most 500 detections per image are scored (hence maxDets 500).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "multiple-object-video-detection-hw-2-ucu-ai-2026"
CONFIGS_DIR = REPO_ROOT / "configs"
SPLITS_DIR = CONFIGS_DIR / "splits"
PROTOCOL_PATH = CONFIGS_DIR / "protocol.json"

SEED = 42

# Label map. COCO category_id 1..8 <-> submission/YOLO class 0..7.
CLASS_NAMES = ("person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor")
NUM_CLASSES = len(CLASS_NAMES)
CATEGORY_IDS = tuple(range(1, NUM_CLASSES + 1))
VISDRONE_ORIG_TO_CATEGORY = {1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 6, 9: 7, 10: 8}
VISDRONE_IGNORE_ORIG = (0, 11)

# ImageNet mean (0.485, 0.456, 0.406) * 255, rounded. Used to fill ignore regions in training images.
IGNORE_FILL_RGB = (124, 116, 104)

DATA_SPLITS = ("train", "val")
FRAME_SPLITS = ("train_rubric", "val_fast", "val_full")


def category_to_class(category_id: int) -> int:
    """COCO category_id (1–8) -> submission / YOLO class (0–7)."""
    if category_id not in CATEGORY_IDS:
        raise ValueError(f"category_id must be in 1..{NUM_CLASSES}, got {category_id}")
    return category_id - 1


def class_to_category(class_id: int) -> int:
    """Submission / YOLO class (0–7) -> COCO category_id (1–8)."""
    if not 0 <= class_id < NUM_CLASSES:
        raise ValueError(f"class_id must be in 0..{NUM_CLASSES - 1}, got {class_id}")
    return class_id + 1


# ---------------------------------------------------------------- paths


def _root(root: str | Path | None) -> Path:
    return Path(root) if root is not None else DATASET_ROOT


def split_dir(split: str, root: str | Path | None = None) -> Path:
    if split not in DATA_SPLITS + ("test",):
        raise ValueError(f"unknown data split {split!r}")
    return _root(root) / split


def coco_path(split: str, root: str | Path | None = None) -> Path:
    return split_dir(split, root) / "annotations_coco.json"


def image_path(split: str, image: dict, root: str | Path | None = None) -> Path:
    """Absolute path of a COCO image record (`file_name` is relative to the split folder)."""
    return split_dir(split, root) / image["file_name"]


def ignore_path(split: str, image: dict, root: str | Path | None = None) -> Path:
    """Path of the frame's ignore txt file. The file is absent when the frame has no ignore region."""
    file_name = Path(image["file_name"])
    return split_dir(split, root) / "ignore" / file_name.parent.name / f"{file_name.stem}.txt"


def clip_name(image: dict) -> str:
    return image["video"]


# ---------------------------------------------------------------- loaders


@lru_cache(maxsize=4)
def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_coco(split: str, root: str | Path | None = None) -> dict:
    """Load `<split>/annotations_coco.json`. The result is cached and shared, so do not mutate it."""
    return _read_json(str(coco_path(split, root)))


def split_annotations(coco: dict) -> dict[int, dict]:
    """Per image id: `{"targets": [ann, ...], "ignore": [[x, y, w, h], ...]}`.

    Targets are the non-crowd annotations. The 8 per-category copies of each crowd region collapse into
    one ignore box. Every image of the COCO file gets an entry, even when it is empty.
    """
    index = {img["id"]: {"targets": [], "ignore": []} for img in coco["images"]}
    seen = set()
    for ann in coco["annotations"]:
        entry = index[ann["image_id"]]
        if ann.get("iscrowd", 0):
            key = (ann["image_id"], tuple(ann["bbox"]))
            if key not in seen:
                seen.add(key)
                entry["ignore"].append(list(ann["bbox"]))
        else:
            entry["targets"].append(ann)
    return index


def read_ignore_file(path: str | Path, width: int, height: int) -> list[list[float]]:
    """Read an ignore txt file (`cx cy w h` normalised) as `[x, y, w, h]` boxes in pixels.

    A missing file means the frame has no ignore regions.
    """
    path = Path(path)
    if not path.exists():
        return []
    boxes = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 4:
            raise ValueError(f"{path}:{line_no}: expected 4 values 'cx cy w h', got {len(values)}")
        cx, cy, w, h = map(float, values)
        boxes.append([(cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height])
    return boxes


def load_test_images(root: str | Path | None = None) -> list[dict]:
    """Rows of `test_images.csv` in file order.

    `image_key` is the CSV `image_id` string (for example `clip_001_0000001`) used in the submission.
    `id` is an integer, the 1-based row number, used as `image_id` in COCO-results JSON for test predictions.
    """
    rows = []
    with open(_root(root) / "test_images.csv", newline="", encoding="utf-8") as f:
        for number, row in enumerate(csv.DictReader(f), 1):
            rows.append({
                "id": number,
                "image_key": row["image_id"],
                "file_name": row["file_name"],
                "clip": row["clip"],
                "frame": int(row["frame"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
            })
    return rows


def test_image_path(row: dict, root: str | Path | None = None) -> Path:
    """`file_name` in `test_images.csv` is relative to the dataset root (`test/images/...`)."""
    return _root(root) / row["file_name"]


# ---------------------------------------------------------------- frozen splits and protocol


def load_protocol() -> dict:
    return _read_json(str(PROTOCOL_PATH))


def load_split(name: str) -> dict:
    """Load a frozen frame split from `configs/splits/<name>.json`."""
    if name not in FRAME_SPLITS:
        raise ValueError(f"unknown frame split {name!r}; expected one of {FRAME_SPLITS}")
    return _read_json(str(SPLITS_DIR / f"{name}.json"))


def split_image_ids(name: str) -> list[int]:
    return list(load_split(name)["image_ids"])


def _frames_by_clip(images: Iterable[dict]) -> dict[str, list[dict]]:
    clips = defaultdict(list)
    for image in images:
        clips[clip_name(image)].append(image)
    return {clip: sorted(frames, key=lambda im: im["frame_id"]) for clip, frames in sorted(clips.items())}


def evenly_spaced_per_clip(images: Iterable[dict], per_clip: int) -> list[int]:
    """Image ids of `per_clip` evenly spaced frames from every clip (legacy notebook rule, `np.linspace`)."""
    ids = []
    for frames in _frames_by_clip(images).values():
        positions = np.linspace(0, len(frames) - 1, min(per_clip, len(frames)), dtype=int)
        ids.extend(frames[p]["id"] for p in positions)
    return ids


def every_kth_per_clip(images: Iterable[dict], k: int) -> list[int]:
    """Image ids of every k-th frame of every clip, starting from the first frame."""
    ids = []
    for frames in _frames_by_clip(images).values():
        ids.extend(im["id"] for im in frames[::k])
    return ids


def clip_overlap(train_coco: dict, val_coco: dict) -> set[str]:
    """Clip names present in both splits (must be empty)."""
    train = {v["name"] for v in train_coco["videos"]}
    val = {v["name"] for v in val_coco["videos"]}
    return train & val


# ---------------------------------------------------------------- ignore policy (training side)


def _pixel_bounds(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x0 = max(0, int(np.floor(x)))
    y0 = max(0, int(np.floor(y)))
    x1 = min(width, int(np.ceil(x + w)))
    y1 = min(height, int(np.ceil(y + h)))
    return x0, y0, x1, y1


def ignore_mask(
    height: int,
    width: int,
    ignore_boxes: Iterable[Sequence[float]],
    keep_boxes: Iterable[Sequence[float]] = (),
) -> np.ndarray:
    """Boolean HxW mask of the pixels to fill: inside an ignore box and outside every keep (target) box."""
    mask = np.zeros((height, width), dtype=bool)
    for box in ignore_boxes:
        x0, y0, x1, y1 = _pixel_bounds(box, width, height)
        mask[y0:y1, x0:x1] = True
    for box in keep_boxes:
        x0, y0, x1, y1 = _pixel_bounds(box, width, height)
        mask[y0:y1, x0:x1] = False
    return mask


def mask_ignore_regions(
    image: np.ndarray,
    ignore_boxes: Iterable[Sequence[float]],
    keep_boxes: Iterable[Sequence[float]] = (),
    fill: Sequence[int] = IGNORE_FILL_RGB,
) -> np.ndarray:
    """Return a copy of an HxWx3 RGB image with the ignore regions filled (training policy).

    Boxes are `[x, y, w, h]` in pixels. Pixels inside `keep_boxes` (the labelled targets) are not changed.
    """
    out = image.copy()
    mask = ignore_mask(image.shape[0], image.shape[1], ignore_boxes, keep_boxes)
    out[mask] = np.asarray(fill, dtype=out.dtype)
    return out


def target_arrays(targets: Iterable[dict], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Target annotations -> (`boxes` Nx4 float32 xyxy clipped to the image, `labels` N int64 category ids 1–8).

    Degenerate boxes (zero width or height after clipping) are dropped.
    Torchvision uses category ids directly (0 = background); YOLO uses `category_to_class`.
    """
    boxes, labels = [], []
    for ann in targets:
        x, y, w, h = ann["bbox"]
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(width), x + w), min(float(height), y + h)
        if x1 > x0 and y1 > y0:
            boxes.append([x0, y0, x1, y1])
            labels.append(ann["category_id"])
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(labels, dtype=np.int64),
    )


def training_sample(
    split: str,
    image: dict,
    entry: dict,
    root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one frame with the frozen training treatment applied.

    `entry` is the frame's item from `split_annotations`. Returns the masked RGB uint8 image,
    the boxes (xyxy pixels) and the labels (category ids 1–8).
    """
    import cv2  # local import: the rest of the module works without OpenCV

    path = image_path(split, image, root)
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    if entry["ignore"]:
        keep = [ann["bbox"] for ann in entry["targets"]]
        rgb = mask_ignore_regions(rgb, entry["ignore"], keep)
    boxes, labels = target_arrays(entry["targets"], width, height)
    return rgb, boxes, labels
