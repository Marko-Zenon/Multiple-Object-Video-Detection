"""OPT-03: tools to study Model A's CH-13 pipeline from raw per-view caches (CPU only).

* `combine_traced` runs exactly the CH-13 merge (`movd.infer.combine`: decode every view top-300 / conf 0.001,
  class-wise NMS, global top 500) but also returns, for every kept box, the view it came from and its rank before
  the cap, so the full / flip / tile contributions and the effect of the cap can be measured.
* `match_frame` is a simple diagnostic matcher at one IoU (greedy by score, per class; crowd regions ignore by
  intersection-over-detection-area like pycocotools). It labels boxes TP / FP / ignored and GT hit / missed.
  It is only for the diagnosis tables; every mAP number comes from `movd.evaluate`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.ops import batched_nms

from movd import infer

VIEWS = ("full", "flip", "tile0", "tile1", "tile2", "tile3")


@dataclass
class Traced:
    """Merged detections of one frame, sorted by score, before the cap; `src` = view index into VIEWS."""

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    src: np.ndarray
    n_candidates: int  # boxes entering NMS (all views)

    def dets(self, max_det: int | None = 500) -> infer.Dets:
        n = len(self.scores) if max_det is None else max_det
        return infer.Dets(self.boxes[:n], self.scores[:n], self.classes[:n])


def combine_traced(
    raws: Sequence[infer.Raw],
    iou: float = 0.5,
    k: int = 300,
    conf: float = 0.001,
    keep_views: Sequence[int] | None = None,
) -> Traced:
    """CH-13 merge without the final cap, with the source view of every kept box.

    `keep_views` restricts the merge to some views (ablations). The result's first 500 boxes equal
    `infer.combine(raws, final_inference config)` when all 6 views are kept (checked in tests).
    """
    views = range(len(raws)) if keep_views is None else keep_views
    parts, srcs = [], []
    for v in views:
        d = infer.decode(raws[v], k, conf)
        parts.append(d)
        srcs.append(np.full(len(d), v, np.int64))
    dets = infer.concat(parts)
    src = np.concatenate(srcs) if srcs else np.zeros(0, np.int64)
    if not len(dets):
        return Traced(dets.boxes, dets.scores, dets.classes, src, 0)
    keep = batched_nms(
        torch.from_numpy(np.ascontiguousarray(dets.boxes, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(dets.scores, dtype=np.float32)),
        torch.from_numpy(dets.classes),
        iou,
    ).numpy()
    # batched_nms returns kept indices sorted by score; the cap is stable on ties, as in infer.cap.
    order = keep[np.argsort(-dets.scores[keep], kind="stable")]
    return Traced(
        dets.boxes[order],
        dets.scores[order],
        dets.classes[order],
        src[order],
        len(dets),
    )


def edge_margins(
    window: tuple[int, int, int, int], width: int, height: int, margin: float
) -> np.ndarray:
    """Per-edge thresholds (x0, y0, x1, y1) of a tile window; inf / -inf for edges that are frame edges."""
    x0, y0, x1, y1 = window
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return np.array(
        [
            x0 + mx if x0 > 0 else -np.inf,
            y0 + my if y0 > 0 else -np.inf,
            x1 - mx if x1 < width else np.inf,
            y1 - my if y1 < height else np.inf,
        ]
    )


def border_mask(boxes: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """True for boxes that reach into the margin of an inner tile edge."""
    t = thresholds
    return (
        (boxes[:, 0] <= t[0])
        | (boxes[:, 1] <= t[1])
        | (boxes[:, 2] >= t[2])
        | (boxes[:, 3] >= t[3])
    )


def edge_filter_combine(
    raws: Sequence[infer.Raw],
    width: int,
    height: int,
    factor: float,
    margin: float,
    config: dict,
) -> infer.Dets:
    """OPT-03 candidate: the CH-13 merge, with tile boxes at an inner tile edge scaled by `factor` (0 drops them).

    Order: decode every view (top-k, conf) -> edge step on the tile views only -> class-wise NMS -> top 500.
    With factor 1 this equals `infer.combine(raws, config)`.
    """
    specs = infer.pass_specs(config)
    conf = config.get("decode", {}).get("conf", 0.001)
    grid, frac = config.get("tile_grid", 2), config.get("tile_frac", 0.6)
    windows = infer.tile_windows(width, height, grid, frac)
    parts = []
    for raw, spec in zip(raws, specs, strict=True):
        d = infer.decode(raw, spec["k"], conf)
        if spec["view"].startswith("tile") and len(d) and factor != 1:
            hit = border_mask(
                d.boxes,
                edge_margins(windows[int(spec["view"][4:])], width, height, margin),
            )
            if factor == 0:
                d = d.take(np.flatnonzero(~hit))
            else:
                d = infer.Dets(
                    d.boxes,
                    np.where(hit, d.scores * factor, d.scores).astype(np.float32),
                    d.classes,
                )
        parts.append(d)
    dets = infer.nms(infer.concat(parts), config["merge"]["iou"])
    return infer.cap(dets, config.get("max_det", 500))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of xyxy boxes a (N, 4) vs b (M, 4)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def ioa_matrix(a: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Intersection over the area of each box in a (pycocotools crowd rule)."""
    if not len(a) or not len(regions):
        return np.zeros((len(a), len(regions)))
    x0 = np.maximum(a[:, None, 0], regions[None, :, 0])
    y0 = np.maximum(a[:, None, 1], regions[None, :, 1])
    x1 = np.minimum(a[:, None, 2], regions[None, :, 2])
    y1 = np.minimum(a[:, None, 3], regions[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    return np.divide(
        inter, area_a[:, None], out=np.zeros_like(inter), where=area_a[:, None] > 0
    )


# Box outcome codes of match_frame.
TP, FP_DUP, FP_CLASS, FP_BG, IGNORED = 0, 1, 2, 3, 4
OUTCOMES = ("tp", "fp_duplicate", "fp_wrong_class", "fp_background", "ignored")


def match_frame(
    boxes: np.ndarray,
    classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    crowd: np.ndarray,
    iou: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy matching of score-sorted detections -> (outcome per box, matched-box index per GT or -1).

    FP sub-types: duplicate (IoU >= iou with an already matched same-class GT), wrong class (IoU >= iou with a
    GT of another class), background (everything else). A non-matching box inside a crowd region
    (intersection / box area >= iou) is ignored.
    """
    outcome = np.full(len(boxes), FP_BG, np.int64)
    gt_hit = np.full(len(gt_boxes), -1, np.int64)
    ious = iou_matrix(boxes, gt_boxes)
    ioa = ioa_matrix(boxes, crowd).max(1) if len(crowd) else np.zeros(len(boxes))
    for i in range(len(boxes)):
        same = gt_classes == classes[i]
        cand = np.where(same & (gt_hit < 0), ious[i], -1.0)
        j = int(np.argmax(cand)) if len(cand) else -1
        if j >= 0 and cand[j] >= iou:
            outcome[i], gt_hit[j] = TP, i
        elif ioa[i] >= iou:
            outcome[i] = IGNORED
        elif len(gt_boxes) and np.any(same & (ious[i] >= iou)):
            outcome[i] = FP_DUP
        elif len(gt_boxes) and np.any(~same & (ious[i] >= iou)):
            outcome[i] = FP_CLASS
    return outcome, gt_hit


def touches_inner_tile_edge(
    boxes: np.ndarray,
    window: tuple[int, int, int, int],
    width: int,
    height: int,
    margin: float = 2.0,
) -> np.ndarray:
    """True for boxes (frame px) that touch an edge of their tile window that is not a frame edge."""
    x0, y0, x1, y1 = window
    hit = np.zeros(len(boxes), bool)
    if x0 > 0:
        hit |= boxes[:, 0] <= x0 + margin
    if y0 > 0:
        hit |= boxes[:, 1] <= y0 + margin
    if x1 < width:
        hit |= boxes[:, 2] >= x1 - margin
    if y1 < height:
        hit |= boxes[:, 3] >= y1 - margin
    return hit
