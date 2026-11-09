"""CH-10 P2 (stride-4) head for YOLO11s: build the starting checkpoint with a documented partial weight transfer.

`configs/models/yolo11s-p2.yaml` is yolo11s (layers 0-22 unchanged) plus an appended P2 branch (layers 23-25)
and a four-level Detect (layer 26, inputs [P3, P4, P5, P2]). The pretrained COCO yolo11s.pt cannot be loaded
by name alone: its Detect head is layer 23, which in the P2 model is the new upsample layer.

`build_p2_init` therefore does the transfer in two steps:

1. Build the plain 8-class yolo11s exactly as the baseline trainer does: `DetectionModel(pretrained.yaml,
   nc=8)`, set the dataset class names (`DetectionTrainer.set_model_names_for_load`), then Ultralytics' own
   `load`. Setting the names first matters: `load` then copies the COCO classification rows of the classes
   that share a name (person, bicycle, car, truck, bus) into the 8-class head. With placeholder names
   it silently skips that and every class row starts untrained (the first CH-10 P2 smoke did this; see
   results/yolo_smoke_p2/README.md).
2. Copy every tensor of that model into the P2 model: layers 0-22 under the same names, and the Detect
   layer 23 -> 26 with the same branch numbers (P3, P4, P5 = branches 0, 1, 2). Every source tensor must
   land on a tensor with the same shape. What stays freshly initialised is exactly the P2 branch
   (layers 23-25 and Detect branch 3).

The result is saved as an Ultralytics checkpoint, so the unchanged `scripts/train_ultra.py` trains it like any
`.pt` (the trainer then reports a full N/N transfer from this file).
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

from movd import data

P2_YAML = data.REPO_ROOT / "configs" / "models" / "yolo11s-p2.yaml"
SOURCE_DETECT = 23  # Detect layer index in yolo11.yaml
P2_DETECT = 26  # Detect layer index in yolo11s-p2.yaml
NEW_LAYERS = (23, 24, 25)  # the appended P2 branch
NEW_DETECT_BRANCH = 3  # Detect input order [P3, P4, P5, P2]


def map_key(key: str) -> str:
    """Name of a plain-yolo11s state-dict key inside the P2 model (only the Detect layer moves)."""
    match = re.match(r"model\.(\d+)\.(.*)", key)
    if not match:
        raise ValueError(f"unexpected state-dict key {key!r}")
    index, rest = int(match.group(1)), match.group(2)
    if index > SOURCE_DETECT:
        raise ValueError(f"{key!r}: yolo11 has no layer {index}")
    return f"model.{P2_DETECT if index == SOURCE_DETECT else index}.{rest}"


def is_new_key(key: str) -> bool:
    """True for P2-model keys with no pretrained counterpart (the P2 branch and Detect branch 3)."""
    index, rest = re.match(r"model\.(\d+)\.(.*)", key).groups()
    if int(index) in NEW_LAYERS:
        return True
    return int(index) == P2_DETECT and bool(
        re.match(rf"(cv2|cv3)\.{NEW_DETECT_BRANCH}\.", rest)
    )


def class_names() -> dict[int, str]:
    """Class 0-7 names (protocol label map), as the trainer sets them from data.yaml."""
    return dict(enumerate(data.CLASS_NAMES))


def remapped_class_rows(model, source) -> dict[str, str]:
    """Target classes whose final cls-conv rows equal a COCO row of the same name on every level (checked)."""
    coco = {str(v).strip().lower(): int(k) for k, v in source.names.items()}
    dst, src = model.model[-1].cv3, source.model[-1].cv3
    rows = {}
    for k, name in model.names.items():
        j = coco.get(name.strip().lower())
        if j is not None and all(
            d[-1].weight[k].equal(s[-1].weight[j].to(d[-1].weight))
            for d, s in zip(dst, src)
        ):
            rows[name] = f"COCO {j} {source.names[j]}"
    return rows


def build_p2_init(
    pretrained: str | Path, out: str | Path, seed: int | None = None
) -> dict:
    """Write the P2 starting checkpoint to `out`; return the transfer report (counts, key lists, hashes)."""
    import torch
    import ultralytics
    from ultralytics.nn.tasks import DetectionModel, load_checkpoint

    seed = data.load_protocol()["seed"] if seed is None else seed
    pretrained, out = Path(pretrained), Path(out)
    source, _ = load_checkpoint(str(pretrained))
    names = class_names()

    torch.manual_seed(seed)
    base = DetectionModel(source.yaml, nc=len(names), verbose=False)
    base.names = names  # as DetectionTrainer.set_model_names_for_load, so load() remaps cls rows by name
    base.load(source, verbose=False)
    matched = remapped_class_rows(base, source)
    if not matched:
        raise ValueError("no COCO classification rows were remapped by class name")
    base_sd = base.state_dict()

    torch.manual_seed(seed)
    p2 = DetectionModel(str(P2_YAML), nc=len(names), verbose=False)
    p2_sd = p2.state_dict()

    copied, problems = {}, []
    for key, value in base_sd.items():
        target = map_key(key)
        if target not in p2_sd:
            problems.append(f"{key} -> {target}: missing in P2 model")
        elif p2_sd[target].shape != value.shape:
            problems.append(
                f"{key} -> {target}: shape {tuple(value.shape)} vs {tuple(p2_sd[target].shape)}"
            )
        else:
            copied[target] = value
    if problems:
        raise ValueError("P2 weight transfer mismatch: " + "; ".join(problems[:5]))
    fresh = sorted(k for k in p2_sd if k not in copied)
    unexpected = [k for k in fresh if not is_new_key(k)]
    if unexpected:
        raise ValueError(
            f"P2 keys left untransferred outside the P2 branch: {unexpected[:5]}"
        )
    p2.load_state_dict({**p2_sd, **copied}, strict=True)
    p2.names = names

    def count(keys):
        return int(sum(p2_sd[k].numel() for k in keys))

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": p2,
            "train_args": {},
            "date": datetime.now(UTC).isoformat(),
            "version": ultralytics.__version__,
            "note": "CH-10 P2 starting checkpoint built by movd/yolo_p2.py (not trained)",
        },
        out,
    )
    strides = [int(s) for s in p2.stride.tolist()]
    return {
        "pretrained": pretrained.name,
        "pretrained_sha256": hashlib.sha256(pretrained.read_bytes()).hexdigest(),
        "model_yaml": P2_YAML.relative_to(data.REPO_ROOT).as_posix(),
        "model_yaml_sha256": hashlib.sha256(P2_YAML.read_bytes()).hexdigest(),
        "init_checkpoint": out.name,
        "init_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "seed": seed,
        "ultralytics": ultralytics.__version__,
        "step1_baseline_build": "DetectionModel(yolo11s.yaml, nc=8), names = data.yaml names, .load(yolo11s.pt): the baseline trainer's path",
        "cls_rows_copied_from_coco_by_name": matched,
        "detect_strides": strides,
        "detect_input_channels": [int(m[0].conv.in_channels) for m in p2.model[-1].cv2],
        "baseline_tensors": len(base_sd),
        "p2_tensors": len(p2_sd),
        "transferred_tensors": len(copied),
        "fresh_tensors": len(fresh),
        "baseline_parameters": int(sum(v.numel() for v in base.parameters())),
        "p2_parameters": int(sum(v.numel() for v in p2.parameters())),
        "p2_state_elements": count(p2_sd),
        "transferred_elements": count(copied),
        "fresh_elements": count(fresh),
        "renamed": "model.23.* (yolo11 Detect) -> model.26.* with the same branch numbers; layers 0-22 same names",
        "fresh_keys": fresh,
    }
