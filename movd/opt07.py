"""OPT-07: predeclared temporal post-processing of cached per-frame detections (CPU, deterministic).

Every function takes COCO-results predictions (`image_id`, `category_id`, `bbox` = x, y, w, h, `score`) and a
neighbour table `{image_id: (previous image_id or None, next image_id or None)}` built from the clip and frame
number of each image, and returns a new list. Inputs are never modified. Matching is always within one class and
between adjacent frames of the same clip only. Support is always computed from the original (unmodified) scores,
so the result does not depend on processing order.

The frozen parameters of each candidate are in `EXPERIMENTS` (rules: results/opt-07-overnight/<id>/rule.md).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

MAX_DET = 500

# Frozen candidates (written before any of them was evaluated; never changed after a result).
EXPERIMENTS: dict[str, dict] = {
    "x1_isolation_demotion": {
        "kind": "isolation_demotion",
        "iou": 0.3,  # a neighbour box supports b if same class and IoU >= 0.3 ...
        "rel_score": 0.5,  # ... and its score >= 0.5 * score(b)
        "factor": 0.5,  # unsupported in every existing adjacent frame: score * 0.5
    },
    "x2_persistence_boost": {
        "kind": "persistence_boost",
        "iou": 0.5,  # best same-class IoU >= 0.5 in BOTH t-1 and t+1
        # score = max(score, mean(score_prev_match, score, score_next_match)); never lowered
    },
    "x3_interpolation": {
        "kind": "interpolation",
        "pair_iou": 0.5,  # same-class t-1 / t+1 boxes, one-to-one greedy by IoU, IoU >= 0.5
        "min_score": 0.3,  # both endpoint scores >= 0.3
        "exist_iou": 0.5,  # skip if any same-class box already in t has IoU >= 0.5 with the midpoint box
        "factor": 0.5,  # new score = 0.5 * min(endpoint scores)
        "max_det": MAX_DET,  # then per-frame top 500 by score
    },
    # OPT-03 e2 unchanged (scripts/opt03_edge.py CANDIDATES["e2"]), confirmed on val_full: tile-view boxes within
    # 0.005 of the tile side from an inner tile edge are dropped before the CH-13 merge. Runs on the Model A raw
    # per-view cache (scripts/opt07_loop.py), not on the merged predictions, so `apply` does not handle it.
    "x4_border_confirmation": {"kind": "opt03_edge", "factor": 0.0, "margin": 0.005},
}


# ---------------------------------------------------------------- frame tables


def neighbour_table(
    frames: Iterable[tuple[int, str, int]],
) -> dict[int, tuple[int | None, int | None]]:
    """`frames` = (image_id, clip, frame number). Neighbours are frame - 1 and frame + 1 of the same clip."""
    by_key = {(clip, frame): image_id for image_id, clip, frame in frames}
    return {
        image_id: (by_key.get((clip, frame - 1)), by_key.get((clip, frame + 1)))
        for (clip, frame), image_id in by_key.items()
    }


def val_neighbours(gt: dict) -> dict[int, tuple[int | None, int | None]]:
    return neighbour_table(
        (im["id"], im["video"], int(im["frame_id"])) for im in gt["images"]
    )


def test_neighbours(rows: Sequence[dict]) -> dict[int, tuple[int | None, int | None]]:
    """`rows` from `movd.data.load_test_images` (integer `id`, `clip`, `frame`)."""
    return neighbour_table((r["id"], r["clip"], r["frame"]) for r in rows)


# ---------------------------------------------------------------- arrays


class Table:
    """Column view of the predictions, grouped by (image_id, category_id)."""

    def __init__(self, preds: Sequence[dict]):
        self.preds = preds
        n = len(preds)
        self.image = np.fromiter(
            (p["image_id"] for p in preds), dtype=np.int64, count=n
        )
        self.cat = np.fromiter(
            (p["category_id"] for p in preds), dtype=np.int64, count=n
        )
        self.score = np.fromiter((p["score"] for p in preds), dtype=np.float64, count=n)
        xywh = np.array([p["bbox"] for p in preds], dtype=np.float64).reshape(n, 4)
        self.xyxy = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], axis=1)
        order = np.lexsort((np.arange(n), self.cat, self.image))
        keys = self.image[order] * 100 + self.cat[order]
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        ends = np.r_[starts[1:], n]
        self.groups = {
            (int(self.image[order[s]]), int(self.cat[order[s]])): order[s:e]
            for s, e in zip(starts, ends, strict=True)
        }

    def get(self, image_id: int | None, cat: int) -> np.ndarray:
        if image_id is None:
            return np.empty(0, dtype=np.int64)
        return self.groups.get((image_id, cat), np.empty(0, dtype=np.int64))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every xyxy box in `a` (n, 4) with every box in `b` (m, 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def _rescored(preds: Sequence[dict], scores: np.ndarray) -> list[dict]:
    return [{**p, "score": float(s)} for p, s in zip(preds, scores, strict=True)]


# ---------------------------------------------------------------- the three rules


def isolation_demotion(
    preds: Sequence[dict], nbrs: dict, iou: float, rel_score: float, factor: float
) -> list[dict]:
    """Multiply by `factor` the score of every box with no support in any existing adjacent frame.

    Support: a same-class box in frame t-1 or t+1 with IoU >= `iou` and score >= `rel_score` * own score.
    A frame without any adjacent frame (a one-frame clip) is left unchanged.
    """
    t = Table(preds)
    supported = np.zeros(len(preds), dtype=bool)
    has_neighbour = np.zeros(len(preds), dtype=bool)
    for (image_id, cat), idx in t.groups.items():
        for other in nbrs.get(image_id, (None, None)):
            if other is None:
                continue
            has_neighbour[idx] = True
            jdx = t.get(other, cat)
            if len(jdx) == 0:
                continue
            m = iou_matrix(t.xyxy[idx], t.xyxy[jdx]) >= iou
            m &= t.score[jdx][None, :] >= rel_score * t.score[idx][:, None]
            supported[idx] |= m.any(axis=1)
    scores = np.where(has_neighbour & ~supported, t.score * factor, t.score)
    return _rescored(preds, scores)


def persistence_boost(preds: Sequence[dict], nbrs: dict, iou: float) -> list[dict]:
    """Raise to the 3-frame mean the score of boxes matched in both t-1 and t+1 (best same-class IoU >= `iou`)."""
    t = Table(preds)
    scores = t.score.copy()
    for (image_id, cat), idx in t.groups.items():
        prev, nxt = nbrs.get(image_id, (None, None))
        if prev is None or nxt is None:
            continue
        side = []
        for other in (prev, nxt):
            jdx = t.get(other, cat)
            if len(jdx) == 0:
                break
            m = iou_matrix(t.xyxy[idx], t.xyxy[jdx])
            best = m.argmax(axis=1)  # first maximum: deterministic
            ok = m[np.arange(len(idx)), best] >= iou
            side.append((ok, t.score[jdx][best]))
        if len(side) < 2:
            continue
        (ok_p, s_p), (ok_n, s_n) = side
        both = ok_p & ok_n
        mean = (s_p + t.score[idx] + s_n) / 3.0
        scores[idx] = np.where(both, np.maximum(t.score[idx], mean), t.score[idx])
    return _rescored(preds, scores)


def _greedy_pairs(m: np.ndarray, thr: float) -> list[tuple[int, int]]:
    """One-to-one pairs by IoU, highest first (ties: row-major order), IoU >= thr."""
    rows, cols = np.nonzero(m >= thr)
    order = np.lexsort((cols, rows, -m[rows, cols]))
    used_r, used_c, pairs = set(), set(), []
    for k in order:
        r, c = int(rows[k]), int(cols[k])
        if r not in used_r and c not in used_c:
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c))
    return pairs


def interpolation(
    preds: Sequence[dict],
    nbrs: dict,
    pair_iou: float,
    min_score: float,
    exist_iou: float,
    factor: float,
    max_det: int,
) -> list[dict]:
    """Add the midpoint of confident same-class t-1 / t+1 pairs that frame t has no box for; then top `max_det`."""
    t = Table(preds)
    added: list[dict] = []
    for image_id in sorted(nbrs):
        prev, nxt = nbrs[image_id]
        if prev is None or nxt is None:
            continue
        cats = sorted({c for (i, c) in t.groups if i in (prev, nxt)})
        for cat in cats:
            p = t.get(prev, cat)
            n = t.get(nxt, cat)
            p = p[t.score[p] >= min_score]
            n = n[t.score[n] >= min_score]
            if len(p) == 0 or len(n) == 0:
                continue
            for r, c in _greedy_pairs(iou_matrix(t.xyxy[p], t.xyxy[n]), pair_iou):
                box = (t.xyxy[p[r]] + t.xyxy[n[c]]) / 2.0
                here = t.get(image_id, cat)
                if (
                    len(here)
                    and (iou_matrix(box[None, :], t.xyxy[here]) >= exist_iou).any()
                ):
                    continue
                added.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(cat),
                        "bbox": [
                            float(box[0]),
                            float(box[1]),
                            float(box[2] - box[0]),
                            float(box[3] - box[1]),
                        ],
                        "score": float(factor * min(t.score[p[r]], t.score[n[c]])),
                    }
                )
    return cap(list(preds) + added, max_det)


def cap(preds: Sequence[dict], max_det: int) -> list[dict]:
    """Per-image top `max_det` by score; ties keep input order (stable)."""
    by_image: dict[int, list[int]] = {}
    for k, p in enumerate(preds):
        by_image.setdefault(p["image_id"], []).append(k)
    keep = []
    for ks in by_image.values():
        ks.sort(key=lambda k: -preds[k]["score"])
        keep.extend(ks[:max_det])
    keep.sort()
    return [dict(preds[k]) for k in keep]


def apply(name: str, preds: Sequence[dict], nbrs: dict) -> list[dict]:
    params = dict(EXPERIMENTS[name])
    kind = params.pop("kind")
    fn = {
        "isolation_demotion": isolation_demotion,
        "persistence_boost": persistence_boost,
        "interpolation": interpolation,
    }[kind]
    return fn(preds, nbrs, **params)
