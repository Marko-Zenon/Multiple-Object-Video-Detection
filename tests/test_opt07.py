"""OPT-07 temporal post-processing rules (synthetic frames, no dataset needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from movd import opt07

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def box(image_id, x, score, cat=1, y=0.0, w=10.0, h=10.0):
    return {
        "image_id": image_id,
        "category_id": cat,
        "bbox": [float(x), y, w, h],
        "score": score,
    }


# three consecutive frames 1, 2, 3 of clip "a" and a one-frame clip "b" (image 9)
NBRS = opt07.neighbour_table([(1, "a", 1), (2, "a", 2), (3, "a", 3), (9, "b", 1)])


def test_neighbour_table():
    assert NBRS == {1: (None, 2), 2: (1, 3), 3: (2, None), 9: (None, None)}


def test_frozen_parameters():
    assert opt07.EXPERIMENTS["x1_isolation_demotion"] == {
        "kind": "isolation_demotion",
        "iou": 0.3,
        "rel_score": 0.5,
        "factor": 0.5,
    }
    assert opt07.EXPERIMENTS["x2_persistence_boost"] == {
        "kind": "persistence_boost",
        "iou": 0.5,
    }
    assert opt07.EXPERIMENTS["x3_interpolation"] == {
        "kind": "interpolation",
        "pair_iou": 0.5,
        "min_score": 0.3,
        "exist_iou": 0.5,
        "factor": 0.5,
        "max_det": 500,
    }


def test_iou_matrix():
    a = opt07.Table([box(1, 0, 0.5), box(1, 5, 0.5)]).xyxy
    m = opt07.iou_matrix(a, a)
    assert m[0, 0] == pytest.approx(1.0)
    assert m[0, 1] == pytest.approx(50 / 150)


def test_isolation_demotion():
    preds = [
        box(2, 0, 0.8),  # supported by frame 1 (same place, score 0.5 >= 0.4)
        box(1, 1, 0.5),  # supported by frame 2
        box(2, 100, 0.8),  # neighbour too weak (0.3 < 0.4): demoted
        box(3, 100, 0.3),
        box(2, 200, 0.8, cat=2),  # neighbour of another class: demoted
        box(1, 200, 0.9, cat=1),  # no neighbour box: demoted
        box(9, 0, 0.7),  # one-frame clip: unchanged
    ]
    out = opt07.apply("x1_isolation_demotion", preds, NBRS)
    assert [p["score"] for p in out] == [0.8, 0.5, 0.4, 0.3, 0.4, 0.45, 0.7]
    assert preds[2]["score"] == 0.8  # input untouched
    assert [p["bbox"] for p in out] == [p["bbox"] for p in preds]


def test_persistence_boost():
    preds = [
        box(1, 0, 0.9),
        box(2, 0, 0.3),
        box(3, 1, 0.6),  # matched both sides: max(0.3, 0.6)
        box(2, 100, 0.9),
        box(1, 100, 0.1),
        box(3, 100, 0.2),  # mean lower than own: unchanged
        box(2, 300, 0.2),
        box(1, 300, 0.9),
    ]  # only one side: unchanged

    out = opt07.apply("x2_persistence_boost", preds, NBRS)
    s = [p["score"] for p in out]
    assert s[1] == pytest.approx(0.6)
    assert s[3] == 0.9 and s[6] == 0.2
    assert s[0] == 0.9 and s[2] == 0.6  # first and last frames have one neighbour only


def test_interpolation_adds_midpoint_and_caps():
    preds = [
        box(1, 0, 0.8),
        box(3, 2, 0.6),  # frame 2 has no box: midpoint x=1, score 0.3
        box(1, 100, 0.8),
        box(3, 100, 0.6),
        box(2, 101, 0.01),  # frame 2 already has one: skipped
        box(1, 200, 0.2),
        box(3, 200, 0.9),
    ]  # endpoint below 0.3: skipped

    out = opt07.apply("x3_interpolation", preds, NBRS)
    new = [p for p in out if p not in preds]
    assert new == [
        {"image_id": 2, "category_id": 1, "bbox": [1.0, 0.0, 10.0, 10.0], "score": 0.3}
    ]
    capped = opt07.cap([box(2, i, 0.1 * i) for i in range(5)], 2)
    assert [p["score"] for p in capped] == [0.30000000000000004, 0.4]


def test_deterministic():
    preds = [
        box(i, (i * 7) % 13, 0.1 + 0.05 * (i % 5), cat=1 + i % 2) for i in (1, 2, 3)
    ] * 3
    for name, params in opt07.EXPERIMENTS.items():
        if params["kind"] == "opt03_edge":
            continue  # runs on the raw cache, not on merged predictions
        assert opt07.apply(name, preds, NBRS) == opt07.apply(name, list(preds), NBRS)


def test_gate():
    import opt07_loop as loop

    def sub(m, small=0.15, dense=0.35):
        return {"mAP": m, "AP_small": small, "density_ge100": dense}

    clips = {f"clip:c{i}": sub(0.3) for i in range(7)}
    base = {
        "val_full": sub(0.30),
        "width_le_1920": sub(0.30),
        "without_4k_clip": sub(0.30),
        **clips,
    }
    good = {
        "val_full": sub(0.301),
        "width_le_1920": sub(0.301),
        "without_4k_clip": sub(0.301),
        **{k: sub(0.3 + (0.001 if i < 4 else -0.001)) for i, k in enumerate(clips)},
    }
    assert loop.gate(base, good)["passes_P1_P6"]
    bad = {**good, "val_full": sub(0.301, small=0.146)}  # AP_small -0.004
    assert not loop.gate(base, bad)["passes_P1_P6"]
    few = {
        **good,
        **{k: sub(0.299) for k in list(clips)[:4]},
    }  # only 3 clips non-negative
    assert not loop.gate(base, few)["checks"]["P5 >= 4 of 7 clips non-negative"]


def test_x4_is_opt03_e2_unchanged():
    import opt03_edge

    x4 = opt07.EXPERIMENTS["x4_border_confirmation"]
    assert x4 == {"kind": "opt03_edge", "factor": 0.0, "margin": 0.005}
    assert (x4["factor"], x4["margin"]) == opt03_edge.CANDIDATES["e2"]


def test_same_predictions():
    import opt07_loop as loop

    a = [box(1, 0, 0.5), box(1, 3, 0.7), box(2, 0, 0.1)]
    assert loop.same_predictions(a, list(reversed(a)))["identical"]
    b = [box(1, 0, 0.5), box(1, 3.5, 0.7), box(2, 0, 0.1)]
    r = loop.same_predictions(a, b)
    assert not r["identical"] and r["max_abs_diff"] == 0.5
    assert not loop.same_predictions(a, a[:2])["identical"]
