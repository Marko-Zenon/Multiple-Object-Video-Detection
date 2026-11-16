"""OPT-02 two-model ensemble: score weighting, class-wise NMS, global cap, specialist routing, determinism."""

from __future__ import annotations

import json
import random

import pytest

from movd import ensemble


def rec(image_id, cat, score, x=0.0, y=0.0, w=10.0, h=10.0):
    return {
        "image_id": image_id,
        "category_id": cat,
        "bbox": [x, y, w, h],
        "score": score,
    }


def test_score_weighting_applies_only_to_the_weighted_model():
    a = [rec(1, 1, 0.8, x=0)]
    b = [rec(1, 1, 0.9, x=100)]
    out = ensemble.merge_frame([a, b], [1.0, 0.5], iou=0.5)
    assert [r["score"] for r in out] == [0.8, 0.45]
    assert out[0]["bbox"] == [0, 0.0, 10.0, 10.0]  # A first: 0.8 > 0.45


def test_weighting_changes_which_overlapping_box_survives():
    a = [rec(1, 1, 0.8)]
    b = [rec(1, 1, 0.9, x=1.0)]  # IoU about 0.82 with a
    assert ensemble.merge_frame([a, b], [1.0, 1.0], 0.5)[0]["score"] == 0.9
    out = ensemble.merge_frame([a, b], [1.0, 0.75], 0.5)
    assert len(out) == 1 and out[0]["score"] == 0.8 and out[0]["bbox"][0] == 0.0


def test_nms_is_class_wise_and_respects_the_iou_threshold():
    a = [rec(1, 1, 0.9), rec(1, 3, 0.85)]  # same box, different classes: both stay
    b = [rec(1, 1, 0.7, x=3.0)]  # IoU 70 / 130 = 0.538 with the class-1 box
    assert len(ensemble.merge_frame([a, b], [1.0, 1.0], 0.5)) == 2
    assert len(ensemble.merge_frame([a, b], [1.0, 1.0], 0.6)) == 3


def test_global_cap_keeps_the_top_scores_over_both_models():
    a = [rec(1, 1, 0.5 + i / 1000, x=20.0 * i) for i in range(300)]
    b = [rec(1, 4, 0.4 + i / 1000, x=20.0 * i) for i in range(300)]
    out = ensemble.merge_frame([a, b], [1.0, 1.0], 0.5, max_det=500)
    assert len(out) == 500
    scores = [r["score"] for r in out]
    assert scores == sorted(scores, reverse=True)
    assert min(scores) == pytest.approx(sorted([r["score"] for r in a + b])[-500])


def test_specialist_routing_takes_bicycle_from_b_and_the_rest_from_a():
    a = [rec(1, 1, 0.9), rec(1, 2, 0.95, x=50)]
    b = [rec(1, 2, 0.3, x=200), rec(1, 1, 0.99, x=400)]
    routes = ensemble.routes_from_config(
        {"B": [2], "A": [1, 3, 4, 5, 6, 7, 8]}, ["A", "B"]
    )
    out = ensemble.merge_frame([a, b], [1.0, 1.0], 0.5, routes=routes)
    assert [(r["category_id"], r["score"]) for r in out] == [(1, 0.9), (2, 0.3)]
    assert ensemble.routes_from_config(None, ["A", "B"]) == [None, None]


def test_boxes_pass_through_unchanged():
    box = [0.1 + 0.2, 1 / 3, 7.000000000000001, 2.9999999999999996]
    out = ensemble.merge_frame([[rec(1, 1, 0.5, *box)], []], [1.0, 1.0], 0.5)
    assert out[0]["bbox"] == box


def test_ties_are_broken_by_model_then_position():
    a = [rec(1, 1, 0.5, x=0), rec(1, 1, 0.5, x=100)]
    b = [rec(1, 1, 0.5, x=50)]
    out = ensemble.merge_frame([a, b], [1.0, 1.0], 0.5)
    assert [r["bbox"][0] for r in out] == [0, 100, 50]


def test_ensemble_is_deterministic_and_grouped_by_image():
    rng = random.Random(42)
    preds = [
        [
            rec(
                img,
                rng.randint(1, 8),
                rng.random(),
                rng.uniform(0, 200),
                rng.uniform(0, 200),
                rng.uniform(2, 40),
                rng.uniform(2, 40),
            )
            for img in (3, 1, 2)
            for _ in range(400)
        ]
        for _ in range(2)
    ]
    first = ensemble.ensemble(preds, [1.0, 0.75], 0.6)
    second = ensemble.ensemble(json.loads(json.dumps(preds)), [1.0, 0.75], 0.6)
    assert json.dumps(first) == json.dumps(second)
    ids = [r["image_id"] for r in first]
    assert ids == sorted(ids)
    assert all(ids.count(i) <= 500 for i in (1, 2, 3))


def test_merge_frame_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        ensemble.merge_frame([[], []], [1.0], 0.5)
    assert ensemble.merge_frame([[], []], [1.0, 1.0], 0.5) == []
