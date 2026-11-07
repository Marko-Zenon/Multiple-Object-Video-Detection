"""Ultralytics data export and prediction adapter (CH-09). Used by YOLO11 (CH-09/CH-10) and RT-DETR (CH-11).

Ultralytics (https://github.com/ultralytics/ultralytics) is AGPL-3.0 licensed; see README.md.

Export (`export_split`). The shipped `images/` + `labels/` + `data.yaml` are only a format reference:
they ignore the ignore regions. The export applies the CH-02 training policy instead:

* labels come from the COCO targets (`movd.data.split_annotations` + `target_arrays`, crowd never a
  target), as YOLO lines `class cx cy w h` (class = category_id - 1, normalised to the original size);
* a frame with ignore regions is masked with `movd.data.training_sample` (ImageNet-mean fill, labelled
  target boxes kept) and re-encoded as JPEG quality 95; a frame without ignore regions is a byte copy;
* the val list exists only because the Ultralytics trainer always builds a val loader. Its images are
  unmodified byte copies of the originals, and it is never used for a reported number.

Headline and curve numbers never come from Ultralytics' own validator. `predict_coco` runs the model on the
original val images, converts the results to COCO-results JSON (category_id 1-8, `[x, y, w, h]` in original
pixels) and applies the shared 500-per-image cap (`movd.evaluate.cap_per_image`), then `movd.evaluate`
(CH-03) scores them on the unchanged COCO GT.

Layout written by `export_split` under `out_root` (git-ignored `outputs/yolo_data/` by default):

    <split_name>/images/<clip>/<frame>.jpg
    <split_name>/labels/<clip>/<frame>.txt     (Ultralytics maps the last /images/ to /labels/)
    <split_name>.txt                           (one ./relative image path per line)
    <split_name>_manifest.json                 (counts, used by check_export)
    data.yaml                                  (write_data_yaml)
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from movd import data
from movd.evaluate import HEADLINE_MAX_DETS, cap_per_image

JPEG_QUALITY = 95
LABEL_DECIMALS = 6
DEFAULT_EXPORT_ROOT = data.REPO_ROOT / "outputs" / "yolo_data"


# ---------------------------------------------------------------- labels


def yolo_lines(
    boxes_xyxy: np.ndarray, category_ids: np.ndarray, width: int, height: int
) -> list[str]:
    """Pixel xyxy boxes + category ids 1-8 -> YOLO lines `class cx cy w h` (normalised, class 0-7)."""
    lines = []
    for (x0, y0, x1, y1), category_id in zip(
        boxes_xyxy.tolist(), category_ids.tolist()
    ):
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        w, h = (x1 - x0) / width, (y1 - y0) / height
        values = " ".join(f"{v:.{LABEL_DECIMALS}f}" for v in (cx, cy, w, h))
        lines.append(f"{data.category_to_class(int(category_id))} {values}")
    return lines


def parse_yolo_lines(
    text: str, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `yolo_lines`: returns pixel xyxy boxes (Nx4 float64) and category ids 1-8."""
    boxes, categories = [], []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(
                f"line {line_no}: expected 'class cx cy w h', got {line!r}"
            )
        cls = int(values[0])
        cx, cy, w, h = map(float, values[1:])
        boxes.append(
            [
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            ]
        )
        categories.append(data.class_to_category(cls))
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4), np.asarray(
        categories, dtype=np.int64
    )


# ---------------------------------------------------------------- export


def _relative_frame_path(image: dict) -> Path:
    """`images/<clip>/<frame>.jpg` from the COCO `file_name` (relative to the split folder)."""
    file_name = Path(image["file_name"])
    return Path(file_name.parent.name) / file_name.name


def _write_atomic_text(path: Path, text: str) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    temp.replace(path)


def export_split(
    name: str,
    source_split: str,
    image_ids: Sequence[int],
    out_root: str | Path = DEFAULT_EXPORT_ROOT,
    root: str | Path | None = None,
    mask: bool = True,
    progress: bool = True,
) -> dict:
    """Export frames of a COCO split to YOLO format under `out_root/<name>/`.

    `mask=True` applies the frozen CH-02 training treatment (train data). `mask=False` byte-copies the original
    image (the val list for Ultralytics' trainer). Labels are always the non-crowd COCO targets.
    Resumable: a frame whose image and label already exist is not rewritten (files are written atomically).
    Returns and stores a manifest with the per-class label counts.
    """
    import cv2  # local import, as in movd.data

    out_root = Path(out_root)
    split_root = out_root / name
    coco = data.load_coco(source_split, root)
    images = {img["id"]: img for img in coco["images"]}
    entries = data.split_annotations(coco)

    list_lines, class_counts = [], Counter()
    stats = Counter()
    iterator = image_ids
    if progress:
        from tqdm import tqdm

        iterator = tqdm(image_ids, desc=f"export {name}", unit="img")
    for image_id in iterator:
        image, entry = images[image_id], entries[image_id]
        relative = _relative_frame_path(image)
        image_out = split_root / "images" / relative
        label_out = (split_root / "labels" / relative).with_suffix(".txt")
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)

        width, height = image["width"], image["height"]
        boxes, categories = data.target_arrays(entry["targets"], width, height)
        stats["targets_in_coco"] += len(entry["targets"])
        stats["degenerate_targets_dropped"] += len(entry["targets"]) - len(boxes)
        masked = mask and bool(entry["ignore"])
        stats["frames_masked" if masked else "frames_copied"] += 1
        stats["ignore_regions"] += len(entry["ignore"])

        if not (image_out.exists() and label_out.exists()):
            temp = image_out.with_name(image_out.stem + ".tmp" + image_out.suffix)
            if masked:
                rgb, _, _ = data.training_sample(source_split, image, entry, root)
                if rgb.shape[:2] != (height, width):
                    raise ValueError(
                        f"{image['file_name']}: size {rgb.shape[:2]} != COCO {(height, width)}"
                    )
                ok = cv2.imwrite(
                    str(temp),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )
                if not ok:
                    raise OSError(f"cv2.imwrite failed for {temp}")
            else:
                shutil.copyfile(data.image_path(source_split, image, root), temp)
            temp.replace(image_out)
            lines = yolo_lines(boxes, categories, width, height)
            _write_atomic_text(label_out, "".join(line + "\n" for line in lines))

        class_counts.update(data.category_to_class(int(c)) for c in categories)
        list_lines.append("./" + (Path(name) / "images" / relative).as_posix())

    _write_atomic_text(
        out_root / f"{name}.txt", "".join(line + "\n" for line in list_lines)
    )
    manifest = {
        "name": name,
        "source_split": source_split,
        "num_images": len(image_ids),
        "masked": mask,
        "jpeg_quality_masked": JPEG_QUALITY,
        "label_decimals": LABEL_DECIMALS,
        "ignore_fill_rgb": list(data.IGNORE_FILL_RGB),
        "labels_per_class": {str(k): class_counts[k] for k in range(data.NUM_CLASSES)},
        "num_labels": sum(class_counts.values()),
        **dict(stats),
    }
    _write_atomic_text(
        out_root / f"{name}_manifest.json", json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def write_data_yaml(out_root: str | Path, train: str, val: str) -> Path:
    """Write `out_root/data.yaml` pointing at the exported list files (absolute `path`, as Ultralytics needs)."""
    out_root = Path(out_root).resolve()
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(data.CLASS_NAMES))
    text = (
        "# Generated by movd/ultra.py (CH-09). Masked train export, CH-02 ignore policy.\n"
        f"# '{val}' is an unmodified copy used only by the Ultralytics trainer, never for reported numbers.\n"
        f"path: {out_root.as_posix()}\n"
        f"train: {train}.txt\n"
        f"val: {val}.txt\n"
        f"nc: {data.NUM_CLASSES}\n"
        f"names:\n{names}\n"
    )
    path = out_root / "data.yaml"
    _write_atomic_text(path, text)
    return path


def check_export(
    name: str,
    source_split: str,
    image_ids: Sequence[int],
    out_root: str | Path = DEFAULT_EXPORT_ROOT,
    root: str | Path | None = None,
    masked: bool = True,
    pixel_samples: int = 40,
) -> dict:
    """Independent check of an export against the COCO JSON. Raises AssertionError on any mismatch.

    Checks: one image + one label file per frame, the list file, class ids 0-7, normalised coordinates in
    [0, 1] with positive size, per-class label counts equal to the non-degenerate COCO targets, label boxes
    within 0.01 px of the COCO targets, and, for `pixel_samples` masked frames, that the filled pixels are
    close to the fill colour and the kept target pixels close to the original (JPEG tolerance).
    """
    import cv2

    out_root = Path(out_root)
    split_root = out_root / name
    coco = data.load_coco(source_split, root)
    images = {img["id"]: img for img in coco["images"]}
    entries = data.split_annotations(coco)

    listed = (out_root / f"{name}.txt").read_text(encoding="utf-8").splitlines()
    assert len(listed) == len(image_ids), (
        f"list has {len(listed)} lines, expected {len(image_ids)}"
    )

    expected, found = Counter(), Counter()
    max_box_error = 0.0
    duplicate_labels = 0
    masked_ids = []
    for line, image_id in zip(listed, image_ids):
        image, entry = images[image_id], entries[image_id]
        relative = _relative_frame_path(image)
        assert line == "./" + (Path(name) / "images" / relative).as_posix(), (
            f"list order/path mismatch at {line}"
        )
        image_file = split_root / "images" / relative
        label_file = (split_root / "labels" / relative).with_suffix(".txt")
        assert image_file.is_file(), f"missing image {image_file}"
        assert label_file.is_file(), f"missing label {label_file}"

        width, height = image["width"], image["height"]
        boxes, categories = data.target_arrays(entry["targets"], width, height)
        text = label_file.read_text(encoding="utf-8")
        rows = [r.split() for r in text.splitlines() if r.strip()]
        values = np.asarray([[float(v) for v in r] for r in rows]).reshape(-1, 5)
        assert np.all(values[:, 0] == np.round(values[:, 0])), (
            f"{label_file}: non-integer class"
        )
        assert (
            values[:, 0].min(initial=0) >= 0
            and values[:, 0].max(initial=0) <= data.NUM_CLASSES - 1
        ), f"{label_file}: class outside 0..{data.NUM_CLASSES - 1}"
        assert np.all((values[:, 1:] >= 0) & (values[:, 1:] <= 1)), (
            f"{label_file}: coordinate outside [0, 1]"
        )
        assert np.all(values[:, 3:] > 0), f"{label_file}: zero-size box"
        duplicate_labels += len(rows) - len({tuple(r) for r in rows})

        parsed, parsed_categories = parse_yolo_lines(text, width, height)
        assert np.array_equal(parsed_categories, categories), (
            f"{label_file}: classes differ from COCO targets"
        )
        if len(boxes):
            max_box_error = max(max_box_error, float(np.abs(parsed - boxes).max()))
        expected.update(data.category_to_class(int(c)) for c in categories)
        found.update(int(v) for v in values[:, 0])
        if masked and entry["ignore"]:
            masked_ids.append(image_id)
        if not masked:
            original = data.image_path(source_split, image, root)
            assert image_file.stat().st_size == original.stat().st_size, (
                f"{image_file}: not a byte copy"
            )

    assert found == expected, (
        f"label counts {dict(found)} != COCO targets {dict(expected)}"
    )
    assert max_box_error < 0.01, (
        f"label boxes differ from COCO targets by {max_box_error:.4f} px"
    )

    # Pixel check on evenly spaced masked frames.
    pixel_rows = []
    for image_id in (
        [
            masked_ids[i]
            for i in np.linspace(
                0, len(masked_ids) - 1, min(pixel_samples, len(masked_ids)), dtype=int
            )
        ]
        if masked_ids
        else []
    ):
        image, entry = images[image_id], entries[image_id]
        exported = cv2.cvtColor(
            cv2.imread(str(split_root / "images" / _relative_frame_path(image))),
            cv2.COLOR_BGR2RGB,
        )
        original = cv2.cvtColor(
            cv2.imread(str(data.image_path(source_split, image, root))),
            cv2.COLOR_BGR2RGB,
        )
        keep = [ann["bbox"] for ann in entry["targets"]]
        fill_mask = data.ignore_mask(
            image["height"], image["width"], entry["ignore"], keep
        )
        keep_mask = data.ignore_mask(image["height"], image["width"], keep)
        fill_error = np.abs(
            exported[fill_mask].astype(int) - np.asarray(data.IGNORE_FILL_RGB)
        ).mean()
        keep_error = (
            np.abs(
                exported[keep_mask].astype(int) - original[keep_mask].astype(int)
            ).mean()
            if keep_mask.any()
            else 0.0
        )
        pixel_rows.append(
            {
                "image_id": image_id,
                "filled_pixels": int(fill_mask.sum()),
                "fill_mae": float(fill_error),
                "kept_target_mae": float(keep_error),
            }
        )
        assert fill_error < 3.0, (
            f"image {image_id}: filled pixels differ from the fill colour (MAE {fill_error:.2f})"
        )
        assert keep_error < 4.0, (
            f"image {image_id}: target pixels changed (MAE {keep_error:.2f})"
        )

    return {
        "name": name,
        "num_images": len(image_ids),
        "num_labels": sum(found.values()),
        "labels_per_class": {
            data.CLASS_NAMES[k]: found[k] for k in range(data.NUM_CLASSES)
        },
        "class_ids_found": sorted(found),
        "label_counts_match_coco_targets": True,
        "max_label_box_error_px": max_box_error,
        "duplicate_label_lines": duplicate_labels,
        "masked_frames": len(masked_ids),
        "pixel_check": pixel_rows,
        "pixel_check_max_fill_mae": max(
            (r["fill_mae"] for r in pixel_rows), default=None
        ),
        "pixel_check_max_kept_target_mae": max(
            (r["kept_target_mae"] for r in pixel_rows), default=None
        ),
    }


def directory_bytes(path: str | Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


# ---------------------------------------------------------------- prediction adapter


def boxes_to_coco(
    image_id: int, xyxy: np.ndarray, classes: np.ndarray, scores: np.ndarray
) -> list[dict]:
    """Ultralytics-style arrays (xyxy original pixels, class 0-7, confidence) -> COCO-results dicts.

    Boxes are clipped to be non-negative in size; the evaluator rejects negative widths.
    """
    detections = []
    for (x0, y0, x1, y1), cls, score in zip(
        np.asarray(xyxy, dtype=float).tolist(),
        np.asarray(classes).tolist(),
        np.asarray(scores, dtype=float).tolist(),
    ):
        detections.append(
            {
                "image_id": int(image_id),
                "category_id": data.class_to_category(int(cls)),
                "bbox": [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)],
                "score": score,
            }
        )
    return detections


def result_to_coco(result, image_id: int) -> list[dict]:
    """One `ultralytics.engine.results.Results` -> COCO-results dicts (boxes are already in original pixels)."""
    boxes = result.boxes
    return boxes_to_coco(
        image_id,
        boxes.xyxy.cpu().numpy(),
        boxes.cls.cpu().numpy(),
        boxes.conf.cpu().numpy(),
    )


def predict_kwargs(
    imgsz: int, device: str, half: bool = False, max_det: int = HEADLINE_MAX_DETS
) -> dict:
    """Prediction settings for every evaluated Ultralytics run (plan.md CH-09 scope 3)."""
    return {
        "imgsz": imgsz,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": max_det,
        "device": device,
        "quantize": 16
        if half
        else None,  # 8.4.x name for FP16 inference (`half` is deprecated)
        "verbose": False,
        "save": False,
    }


def predict_coco(
    model,
    frames: Iterable[tuple[int, str | Path]],
    imgsz: int,
    device: str,
    half: bool = False,
    batch: int = 1,
    max_det: int = HEADLINE_MAX_DETS,
    progress: bool = True,
    classes: Sequence[int] | None = None,
) -> list[dict]:
    """Run an Ultralytics model on `(image_id, path)` frames (original images) -> capped COCO-results list.

    `classes` filters the model's class ids. It is only for the format preflight of a COCO-pretrained
    (80-class) model; fine-tuned 8-class checkpoints never need it.
    """
    frames = list(frames)
    kwargs = predict_kwargs(imgsz, device, half, max_det)
    if classes is not None:
        kwargs["classes"] = list(classes)
    detections = []
    chunks = range(0, len(frames), batch)
    if progress:
        from tqdm import tqdm

        chunks = tqdm(chunks, desc="predict", unit="batch")
    for start in chunks:
        chunk = frames[start : start + batch]
        results = model.predict([str(path) for _, path in chunk], **kwargs)
        for (image_id, _), result in zip(chunk, results):
            detections.extend(result_to_coco(result, image_id))
    return cap_per_image(detections, max_det)


def val_frames(
    split: str = "val_fast", root: str | Path | None = None
) -> list[tuple[int, Path]]:
    """(COCO image id, original image path) for a frozen val split, in split order."""
    images = {img["id"]: img for img in data.load_coco("val", root)["images"]}
    return [
        (i, data.image_path("val", images[i], root))
        for i in data.split_image_ids(split)
    ]


# ---------------------------------------------------------------- training arguments


def train_kwargs(
    config: dict, data_yaml: str | Path, project: str | Path, device: str
) -> dict:
    """Ultralytics `model.train` arguments from a run config plus the frozen protocol.

    Rubric baselines: horizontal flip only (configs/protocol.json "augmentation_rubric_baselines"), seed 42,
    per-epoch checkpoints, Ultralytics' own validation off (our evaluator makes the curve). Any key under
    `config["overrides"]` is passed through and must be recorded as an experiment variable.
    """
    protocol = data.load_protocol()
    budget = protocol["rubric_budget"]
    kwargs = {
        "data": str(data_yaml),
        "epochs": config.get("epochs", budget["epochs"]),
        "imgsz": config.get("imgsz", budget["input_long_side"]),
        "batch": config["batch_size"],
        "workers": config.get("workers", 2),
        "device": device,
        "seed": protocol["seed"],
        "deterministic": True,
        "amp": config.get("amp", True),
        "pretrained": True,
        "save_period": 1,
        "val": False,
        "plots": False,
        "project": str(Path(project).resolve()),
        "name": "train",
        "exist_ok": True,
        "max_det": HEADLINE_MAX_DETS,
        # Augmentation: flip only.
        "fliplr": 0.5,
        "flipud": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
    }
    kwargs.update(config.get("overrides", {}))
    if config.get("smoke_fraction"):
        kwargs["fraction"] = config["smoke_fraction"]
    return kwargs
