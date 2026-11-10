"""Small-object specialist merge (OPT-05): Model A predictions + only the small boxes of a second model (FB).

Inputs are cached COCO-results predictions (`image_id`, `category_id`, `bbox` = xywh in original pixels, `score`).
For every frame:

1. all Model A records are kept unchanged;
2. FB records are kept only if their predicted box is small under the size rule, and their score is multiplied by a
   fixed multiplier;
3. the union is sorted stably by score (ties: A before FB, then position in the source list);
4. class-wise NMS (torchvision `batched_nms`, as in `movd.infer.nms` and `movd.ensemble`);
5. the global top `max_det` by score.

Output records are copies of the input records (only the FB `score` is replaced), so boxes pass through bit for bit.
The rule is frozen in `results/opt-05-small-specialist/selection_rule.md`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import torch
from torchvision.ops import batched_nms

from movd.ensemble import group_by_image, xywh_to_xyxy

MODEL_A_INPUT = 1344  # Model A sees each frame as a stretched 1344 x 1344 square
SMALL_SIDE = 32  # COCO small: area < 32^2
SIZE_RULES = ("coco32", "in32")


def check_records(records: Iterable[dict], name: str) -> None:
    """Reject malformed predictions before any merge (missing keys, bad box, non-finite score)."""
    for i, r in enumerate(records):
        try:
            box = r["bbox"]
            score = float(r["score"])
            int(r["image_id"]), int(r["category_id"])
        except (KeyError, TypeError, ValueError) as err:
            raise ValueError(f"{name}[{i}]: malformed record {r!r}") from err
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError(f"{name}[{i}]: bbox must be 4 finite numbers, got {box!r}")
        if box[2] < 0 or box[3] < 0:
            raise ValueError(f"{name}[{i}]: negative box size {box!r}")
        if not math.isfinite(score):
            raise ValueError(f"{name}[{i}]: score must be finite, got {score!r}")


def is_small(box: Sequence[float], width: int, height: int, rule: str) -> bool:
    """Size rule on a predicted xywh box in original pixels of a `width` x `height` frame."""
    w, h = float(box[2]), float(box[3])
    if rule == "coco32":
        return w * h < SMALL_SIDE**2
    if rule == "in32":
        return (w * MODEL_A_INPUT / width) * (
            h * MODEL_A_INPUT / height
        ) < SMALL_SIDE**2
    raise ValueError(f"unknown size rule {rule!r}; expected one of {SIZE_RULES}")


def eligible_fb(
    records: Sequence[dict], width: int, height: int, rule: str, multiplier: float
) -> list[dict]:
    """FB records that pass the size rule, with the score multiplied (copies, inputs untouched)."""
    return [
        {**r, "score": float(r["score"]) * float(multiplier)}
        for r in records
        if is_small(r["bbox"], width, height, rule)
    ]


def merge_frame(
    a: Sequence[dict], fb_small: Sequence[dict], iou: float, max_det: int = 500
) -> list[dict]:
    """Union of A and the eligible (already scaled) FB records, class-wise NMS, top `max_det` by score."""
    records = [*a, *fb_small]
    if not records:
        return []
    scores = np.asarray([float(r["score"]) for r in records], dtype=np.float64)
    source = np.asarray([0] * len(a) + [1] * len(fb_small))
    pos = np.asarray([*range(len(a)), *range(len(fb_small))])
    order = np.lexsort(
        (pos, source, -scores)
    )  # score desc, then A before FB, then source position
    records = [records[i] for i in order]
    scores = scores[order]
    keep = batched_nms(
        torch.from_numpy(xywh_to_xyxy(records)),
        torch.from_numpy(scores.astype(np.float32)),
        torch.tensor([r["category_id"] for r in records], dtype=torch.int64),
        iou,
    ).numpy()
    keep = np.sort(keep)[
        :max_det
    ]  # back to the stable sorted order, then the global cap
    return [records[i] for i in keep]


def merge(
    a_predictions: Sequence[dict],
    fb_predictions: Sequence[dict],
    sizes: Mapping[int, tuple[int, int]],
    rule: str,
    multiplier: float,
    iou: float,
    max_det: int = 500,
    image_ids: Iterable[int] | None = None,
) -> list[dict]:
    """Merge whole prediction lists. `sizes[image_id] = (width, height)`. Output sorted by image id, then score."""
    if rule not in SIZE_RULES:
        raise ValueError(f"unknown size rule {rule!r}; expected one of {SIZE_RULES}")
    if iou < 0.5:
        raise ValueError(
            "NMS IoU below 0.5 would let Model A boxes suppress each other (A is NMS-0.5 output)"
        )
    if not multiplier > 0:
        raise ValueError("FB score multiplier must be > 0")
    check_records(a_predictions, "A")
    check_records(fb_predictions, "FB")
    ga, gfb = group_by_image(a_predictions), group_by_image(fb_predictions)
    ids = sorted(set(ga) | set(gfb)) if image_ids is None else sorted(set(image_ids))
    missing = [i for i in ids if i not in sizes]
    if missing:
        raise ValueError(
            f"no image size for {len(missing)} image ids, e.g. {missing[:3]}"
        )
    out: list[dict] = []
    for image_id in ids:
        w, h = sizes[image_id]
        small = eligible_fb(gfb.get(image_id, []), w, h, rule, multiplier)
        out.extend(merge_frame(ga.get(image_id, []), small, iou, max_det))
    return out
