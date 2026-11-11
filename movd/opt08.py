"""OPT-08: predeclared post-processing on top of / as alternatives to the OPT-07 x2 baseline (CPU, deterministic).

Same conventions as `movd.opt07`: COCO-results predictions in, a new list out, inputs never modified, matching only
within one class and one clip, support always computed from the original scores (order independent), boxes never
moved, no box added or removed.

Every candidate grid is frozen in `EXPERIMENTS` before any of its members is evaluated
(rules: results/opt-08-overnight/<experiment>/rule.md). `SPLIT_SALT` fixes the clip-level discovery / confirmation split.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from movd.opt07 import Table, iou_matrix, persistence_boost

SPLIT_SALT = "OPT-08 split v1:"
N_DISCOVERY = 4


def split_clips(clips: Sequence[str]) -> dict[str, list[str]]:
    """Sort clips by sha256(SPLIT_SALT + name); the first N_DISCOVERY are discovery, the rest confirmation."""
    order = sorted(
        sorted(set(clips)),
        key=lambda c: hashlib.sha256((SPLIT_SALT + c).encode()).hexdigest(),
    )
    return {"discovery": order[:N_DISCOVERY], "confirmation": order[N_DISCOVERY:]}


# Frozen candidate grids (written before any of them was evaluated; never changed after a result).
EXPERIMENTS: dict[str, dict[str, dict]] = {
    # E1: OPT-07 x4 (OPT-03 e2 border drop on the raw cache) then OPT-07 x2 (persistence boost, IoU 0.5). No retuning.
    "e1_x4_then_x2": {
        "c1": {"kind": "x4_then_x2", "margin": 0.005, "factor": 0.0, "iou": 0.5},
    },
    # E2: short-tracklet score smoothing. Links = mutual best same-class IoU >= 0.5 between adjacent frames
    # (the x2 matching made one-to-one). Applied only to boxes with a tracklet member on both sides (as x2).
    # Window = tracklet members within +-k frames. Only the aggregation and its strength change.
    "e2_tracklet_smoothing": {
        "t1": {"kind": "tracklet", "agg": "boost_mean", "k": 2},
        "t2": {"kind": "tracklet", "agg": "boost_mean", "k": 3},
        "t3": {"kind": "tracklet", "agg": "boost_mean", "k": 5},
        "t4": {"kind": "tracklet", "agg": "replace_mean", "k": 2},
        "t5": {"kind": "tracklet", "agg": "boost_median", "k": 2},
        "t6": {"kind": "tracklet", "agg": "boost_triangular", "k": 3},
    },
    # E3: broader conservative support for the x2 boost (one axis: which frames may support a box). IoU 0.5,
    # best same-class match per frame (first maximum), boost only (score never lowered), no box added.
    "e3_broader_support": {
        "b1": {"kind": "support", "mode": "endpoint_one_sided"},
        "b2": {"kind": "support", "mode": "gap_tolerant"},
        "b3": {"kind": "support", "mode": "five_frame_mean"},
        "b4": {"kind": "support", "mode": "gap_tolerant_endpoint"},
    },
    # E4: weight soup w * Model A + (1 - w) * epoch1.pt (scripts/opt08_soup.py), unchanged CH-13 pipeline, then x2.
    "e4_checkpoint_soup": {
        "s1": {"kind": "soup", "tag": "w080", "iou": 0.5},
        "s2": {"kind": "soup", "tag": "w065", "iou": 0.5},
        "s3": {"kind": "soup", "tag": "w050", "iou": 0.5},
    },
    # E5: one extra epoch from Model A at AdamW constant lr0 5e-5 (configs/runs/opt08_e5_lowlr.json), its epoch
    # checkpoint through the same raw cache + CH-13 merge as E4, then x2.
    "e5_lowlr_epoch": {
        "l1": {"kind": "soup", "tag": "e5l1", "iou": 0.5},
    },
    # E6 (TASK.md rule, fixed before any E6 result): every component that passed S1-S3 and U1-U8 on its own:
    # E4 w080 weights -> E1's x4 border drop on the soup's raw views -> E3 b3 (best passed temporal rule).
    "e6_combination": {
        "k1": {
            "kind": "combo",
            "tag": "w080",
            "factor": 0.0,
            "margin": 0.005,
            "mode": "five_frame_mean",
        },
    },
}

IOU = 0.5


# ---------------------------------------------------------------- E2: tracklets


def mutual_links(t: Table, nbrs: dict) -> np.ndarray:
    """nxt[i] = index of the box linked to box i in the next frame, or -1.

    Link i -> j when j is i's best same-class IoU match in t+1, i is j's best match in t (first maximum on ties)
    and the IoU >= 0.5. Mutual best matches are one-to-one, so links form simple chains.
    """
    nxt = np.full(len(t.preds), -1, dtype=np.int64)
    for (image_id, cat), idx in t.groups.items():
        following = nbrs.get(image_id, (None, None))[1]
        jdx = t.get(following, cat)
        if len(jdx) == 0:
            continue
        m = iou_matrix(t.xyxy[idx], t.xyxy[jdx])
        fwd = m.argmax(axis=1)
        bwd = m.argmax(axis=0)
        rows = np.arange(len(idx))
        ok = (bwd[fwd] == rows) & (m[rows, fwd] >= IOU)
        nxt[idx[ok]] = jdx[fwd[ok]]
    return nxt


def tracklets(nxt: np.ndarray) -> list[np.ndarray]:
    """Chains of box indices in frame order (every box belongs to exactly one, possibly of length 1)."""
    has_prev = np.zeros(len(nxt), dtype=bool)
    has_prev[nxt[nxt >= 0]] = True
    chains = []
    for head in np.flatnonzero(~has_prev):
        chain = [int(head)]
        while nxt[chain[-1]] >= 0:
            chain.append(int(nxt[chain[-1]]))
        chains.append(np.array(chain, dtype=np.int64))
    return chains


def _aggregate(s: np.ndarray, p: int, agg: str, k: int) -> float:
    lo, hi = max(0, p - k), min(len(s), p + k + 1)
    w = s[lo:hi]
    own = float(s[p])
    if agg == "boost_mean":
        return max(own, float(w.mean()))
    if agg == "replace_mean":
        return float(w.mean())
    if agg == "boost_median":
        return max(own, float(np.median(w)))
    if agg == "boost_triangular":
        d = np.abs(np.arange(lo, hi) - p)
        weights = (k + 1 - d).astype(np.float64)
        return max(own, float((w * weights).sum() / weights.sum()))
    raise ValueError(agg)


def tracklet_smoothing(
    preds: Sequence[dict], nbrs: dict, agg: str, k: int
) -> list[dict]:
    t = Table(preds)
    scores = t.score.copy()
    for chain in tracklets(mutual_links(t, nbrs)):
        if len(chain) < 3:
            continue
        s = t.score[chain]
        for p in range(1, len(chain) - 1):  # a member on both sides, as x2
            scores[chain[p]] = _aggregate(s, p, agg, k)
    return [{**p, "score": float(v)} for p, v in zip(preds, scores, strict=True)]


# ---------------------------------------------------------------- E3: broader support


def _frame_at(nbrs: dict, image_id: int, offset: int) -> int | None:
    cur: int | None = image_id
    step = 0 if offset < 0 else 1
    for _ in range(abs(offset)):
        if cur is None:
            return None
        cur = nbrs.get(cur, (None, None))[step]
    return cur


def _best_match(t: Table, idx: np.ndarray, other: int | None, cat: int):
    """(matched mask, matched score) of each box in idx against frame `other` (x2 logic: best IoU, first max)."""
    jdx = t.get(other, cat)
    if len(jdx) == 0:
        return np.zeros(len(idx), dtype=bool), np.zeros(len(idx))
    m = iou_matrix(t.xyxy[idx], t.xyxy[jdx])
    best = m.argmax(axis=1)
    return m[np.arange(len(idx)), best] >= IOU, t.score[jdx][best]


def _either(near, far):
    (ok_n, s_n), (ok_f, s_f) = near, far
    return ok_n | ok_f, np.where(ok_n, s_n, s_f)


def broader_support(preds: Sequence[dict], nbrs: dict, mode: str) -> list[dict]:
    """score := max(score, mean(own score, supporting scores)) under the support rule `mode`.

    endpoint_one_sided: x2 (matched in t-1 and t+1); where t-1 or t+1 does not exist (clip end), matched in both
        frames on the existing side (t+1, t+2 or t-1, t-2) instead.
    gap_tolerant: each side is supported by its t-+1 match, or if unmatched by its t-+2 match; both sides needed.
    five_frame_mean: x2 condition (t-1 and t+1 matched); the mean also uses the t-2 / t+2 matches when present.
    gap_tolerant_endpoint: gap_tolerant, plus endpoint_one_sided at clip ends.
    """
    t = Table(preds)
    scores = t.score.copy()
    for (image_id, cat), idx in t.groups.items():
        match = {
            d: _best_match(t, idx, _frame_at(nbrs, image_id, d), cat)
            for d in (-2, -1, 1, 2)
        }
        exists = {d: _frame_at(nbrs, image_id, d) is not None for d in (-2, -1, 1, 2)}
        own = t.score[idx]
        total, count = own.copy(), np.ones(len(idx))
        apply = np.zeros(len(idx), dtype=bool)

        if mode in ("endpoint_one_sided", "five_frame_mean"):
            okp, sp = match[-1]
            okn, sn = match[1]
        else:  # each side: the t-+1 match, else the t-+2 match
            okp, sp = _either(match[-1], match[-2])
            okn, sn = _either(match[1], match[2])
        both = okp & okn
        apply |= both
        total += np.where(both, sp + sn, 0.0)
        count += np.where(both, 2, 0)
        if mode == "five_frame_mean":
            for d in (-2, 2):
                ok, s = match[d]
                add = both & ok
                total += np.where(add, s, 0.0)
                count += add
        if mode in ("endpoint_one_sided", "gap_tolerant_endpoint"):
            for near, far, missing in ((1, 2, -1), (-1, -2, 1)):
                if exists[missing] or not exists[far]:
                    continue
                ok1, s1 = match[near]
                ok2, s2 = match[far]
                one = ok1 & ok2 & ~apply
                apply |= one
                total += np.where(one, s1 + s2, 0.0)
                count += np.where(one, 2, 0)
        mean = total / count
        scores[idx] = np.where(apply, np.maximum(own, mean), own)
    return [{**p, "score": float(v)} for p, v in zip(preds, scores, strict=True)]


# ---------------------------------------------------------------- dispatch


def apply(params: dict, preds: Sequence[dict], nbrs: dict) -> list[dict]:
    """Score-only transforms of merged predictions (E1's x4 step runs on the raw cache in the loop script)."""
    kind = params["kind"]
    if kind in ("x4_then_x2", "soup"):
        return persistence_boost(preds, nbrs, params["iou"])
    if kind == "tracklet":
        return tracklet_smoothing(preds, nbrs, params["agg"], params["k"])
    if kind in ("support", "combo"):
        return broader_support(preds, nbrs, params["mode"])
    raise ValueError(kind)
