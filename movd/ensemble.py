"""Two-model detection ensemble on cached COCO-results predictions (OPT-02).

Each model's predictions are the final output of the frozen CH-13 pipeline (already NMS-merged and capped at 500
per frame). For every frame the ensemble:

1. multiplies each model's scores by that model's weight, optionally keeping only some classes per model
   (`routes`, used by the per-class specialist);
2. unions the detections (model order, then cached order) and sorts them stably by descending weighted score;
3. runs class-wise NMS (torchvision `batched_nms`, the same call as `movd.infer.nms`) at one IoU;
4. keeps the top `max_det` by weighted score (stable, like `movd.infer.cap`).

The output records are copies of the cached records with only `score` replaced by the weighted score, so boxes
are passed through bit for bit (no xywh -> xyxy -> xywh round trip). With the same inputs the output is the same
list in the same order (ties are broken by model index, then the record's position in its model's list).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import torch
from torchvision.ops import batched_nms


def group_by_image(predictions: Iterable[dict]) -> dict[int, list[dict]]:
    """image_id -> that image's records, in their original order."""
    out: dict[int, list[dict]] = defaultdict(list)
    for p in predictions:
        out[p["image_id"]].append(p)
    return out


def xywh_to_xyxy(records: Sequence[dict]) -> np.ndarray:
    if not records:
        return np.zeros((0, 4), np.float32)
    b = np.asarray([r["bbox"] for r in records], dtype=np.float64)
    return np.concatenate([b[:, :2], b[:, :2] + b[:, 2:]], axis=1).astype(np.float32)


def merge_frame(
    per_model: Sequence[Sequence[dict]],
    weights: Sequence[float],
    iou: float,
    max_det: int = 500,
    routes: Sequence[set[int] | None] | None = None,
) -> list[dict]:
    """Ensemble one frame. `per_model[m]` are model m's records; `routes[m]` its allowed category ids (None = all)."""
    if len(per_model) != len(weights):
        raise ValueError("one weight per model")
    routes = routes or [None] * len(per_model)
    records, scores, model_ix, pos = [], [], [], []
    for m, (recs, w, allowed) in enumerate(zip(per_model, weights, routes)):
        for i, r in enumerate(recs):
            if allowed is not None and r["category_id"] not in allowed:
                continue
            records.append(r)
            scores.append(float(r["score"]) * float(w))
            model_ix.append(m)
            pos.append(i)
    if not records:
        return []
    scores = np.asarray(scores, dtype=np.float64)
    # stable order: weighted score desc, then model index, then position in the model's list
    order = np.lexsort((np.asarray(pos), np.asarray(model_ix), -scores))
    records = [records[i] for i in order]
    scores = scores[order]
    boxes = torch.from_numpy(xywh_to_xyxy(records))
    classes = torch.tensor([r["category_id"] for r in records], dtype=torch.int64)
    keep = batched_nms(
        boxes, torch.from_numpy(scores.astype(np.float32)), classes, iou
    ).numpy()
    keep = np.sort(keep)[
        :max_det
    ]  # back to the stable pre-sorted order, then the global cap
    return [{**records[i], "score": float(scores[i])} for i in keep]


def ensemble(
    model_predictions: Sequence[Sequence[dict]],
    weights: Sequence[float],
    iou: float,
    max_det: int = 500,
    routes: Sequence[set[int] | None] | None = None,
    image_ids: Iterable[int] | None = None,
) -> list[dict]:
    """Ensemble whole prediction lists; output grouped by image (sorted image ids), each image score-sorted."""
    grouped = [group_by_image(p) for p in model_predictions]
    ids = (
        sorted(set().union(*[g.keys() for g in grouped]))
        if image_ids is None
        else sorted(set(image_ids))
    )
    out: list[dict] = []
    for image_id in ids:
        out.extend(
            merge_frame(
                [g.get(image_id, []) for g in grouped], weights, iou, max_det, routes
            )
        )
    return out


def routes_from_config(route: Mapping[str, Sequence[int]] | None, names: Sequence[str]):
    """candidates.json `route` ({"A": [...], "B": [...]} or null) -> per-model allowed category-id sets."""
    if route is None:
        return [None] * len(names)
    return [set(route.get(n, [])) for n in names]
