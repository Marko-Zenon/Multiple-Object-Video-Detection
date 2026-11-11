"""OPT-09: one focused local search around the protected OPT-08 E6 pipeline (deterministic, CPU post-processing).

E6 = soup w080 (0.8 x Model A epoch0.pt + 0.2 x epoch1.pt) -> CH-13 merge with inner tile-border boxes dropped
(factor 0, margin 0.005 of the tile side) -> OPT-08 b3 five-frame persistence boost (IoU 0.5).

Every OPT-09 candidate changes exactly one of E6's three components. The grids below were frozen before any
of them was evaluated (results/opt-09-e6-tuning/selection_rule.md) and are never changed after a result.
The clip split is OPT-08's (`movd.opt08.split_clips`, results/opt-08-overnight/split.json), reused unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from movd.opt07 import Table, iou_matrix

# The protected E6 settings (OPT-08 e6_combination k1).
E6 = {"soup": "w080", "factor": 0.0, "margin": 0.005, "iou": 0.5, "strength": 1.0}

# Margins already tested with factor 0: OPT-03 e1-e4 (0.002, 0.005, 0.01, 0.02); OPT-07 x4 and OPT-08 E1 / E6 used
# 0.005 only. Soup weights on Model A already tested: 1.0 (Model A), 0.8, 0.65, 0.5 (OPT-08 E4).
TESTED = {
    "margin": (0.002, 0.005, 0.01, 0.02),
    "persistence_iou": (0.5,),
    "persistence_strength": (1.0,),
    "soup_weight_model_A": (1.0, 0.8, 0.65, 0.5),
}

# Frozen grids. Each candidate overrides exactly one E6 key.
GRIDS: dict[str, dict[str, dict]] = {
    # Phase A1: only the inner tile-border margin (boxes still dropped, factor 0).
    "a1_border_margin": {
        "m1": {"margin": 0.0035},
        "m2": {"margin": 0.0065},
        "m3": {"margin": 0.008},
    },
    # Phase A2: the five-frame persistence boost. Sub-grid "strength": only the boost strength lambda;
    # sub-grid "iou": only the matching IoU. Never both.
    "a2_persistence": {
        "p1": {"strength": 0.75},
        "p2": {"strength": 1.25},
        "p3": {"strength": 1.5},
        "p4": {"iou": 0.45},
        "p5": {"iou": 0.55},
    },
    # Phase B: only the soup weight on Model A (w x epoch0.pt + (1 - w) x epoch1.pt); border and boost as E6.
    "b_soup_weight": {
        "w1": {"soup": "w075"},
        "w2": {"soup": "w085"},
        "w3": {"soup": "w090"},
    },
}

SOUP_WEIGHTS = {"w080": 0.8, "w075": 0.75, "w085": 0.85, "w090": 0.9}


def candidate(grid: str, cid: str) -> dict:
    """The full pipeline parameters of one candidate: E6 with one key overridden."""
    change = GRIDS[grid][cid]
    if len(change) != 1 or next(iter(change)) not in E6:
        raise ValueError(f"{grid}/{cid} must change exactly one E6 parameter")
    return {**E6, **change}


def changed_keys(params: dict) -> list[str]:
    return sorted(k for k in E6 if params[k] != E6[k])


def _frame_at(nbrs: dict, image_id: int, offset: int) -> int | None:
    cur: int | None = image_id
    step = 0 if offset < 0 else 1
    for _ in range(abs(offset)):
        if cur is None:
            return None
        cur = nbrs.get(cur, (None, None))[step]
    return cur


def _best_match(t: Table, idx: np.ndarray, other: int | None, cat: int, iou: float):
    """(matched mask, matched score) of each box in idx in frame `other`: best same-class IoU, first maximum."""
    jdx = t.get(other, cat)
    if len(jdx) == 0:
        return np.zeros(len(idx), dtype=bool), np.zeros(len(idx))
    m = iou_matrix(t.xyxy[idx], t.xyxy[jdx])
    best = m.argmax(axis=1)
    return m[np.arange(len(idx)), best] >= iou, t.score[jdx][best]


def five_frame_boost(
    preds: Sequence[dict], nbrs: dict, iou: float = 0.5, strength: float = 1.0
) -> list[dict]:
    """OPT-08 b3 with an explicit matching IoU and boost strength.

    A box is supported when its best same-class match in t-1 and in t+1 both have IoU >= `iou`. Its mean is the
    mean of its own score, the t-1 / t+1 match scores and the t-2 / t+2 match scores where those also match.
    If mean > own: score := own + strength * (mean - own), capped at 1.0. strength 1 is exactly OPT-08 b3
    (max(own, mean)). Scores are never lowered; no box is moved, added or removed. Support always uses the
    original scores, so the result does not depend on the processing order.
    """
    t = Table(preds)
    scores = t.score.copy()
    for (image_id, cat), idx in t.groups.items():
        match = {
            d: _best_match(t, idx, _frame_at(nbrs, image_id, d), cat, iou)
            for d in (-2, -1, 1, 2)
        }
        own = t.score[idx]
        okp, sp = match[-1]
        okn, sn = match[1]
        both = okp & okn
        total = own + np.where(both, sp + sn, 0.0)
        count = np.ones(len(idx)) + np.where(both, 2, 0)
        for d in (-2, 2):
            ok, s = match[d]
            add = both & ok
            total += np.where(add, s, 0.0)
            count += add
        mean = total / count
        if strength == 1.0:
            boosted = np.maximum(own, mean)
        else:
            boosted = np.minimum(1.0, own + strength * np.maximum(0.0, mean - own))
        scores[idx] = np.where(both, boosted, own)
    return [{**p, "score": float(v)} for p, v in zip(preds, scores, strict=True)]
