"""OPT-05 small-object specialist merge (CPU only)."""

from __future__ import annotations

import json

import pytest

from movd import specialist


def rec(image_id, cls, x, y, w, h, score, **extra):
    return {
        "image_id": image_id,
        "category_id": cls,
        "bbox": [x, y, w, h],
        "score": score,
        **extra,
    }


SIZES = {1: (1344, 756), 2: (3840, 2160)}


def test_size_rules():
    # coco32: area < 1024 px^2 in original pixels, independent of the frame size
    assert specialist.is_small([0, 0, 31, 33], 3840, 2160, "coco32")  # 1023
    assert not specialist.is_small(
        [0, 0, 32, 32], 3840, 2160, "coco32"
    )  # 1024, not < 1024
    # in32 on 1344x756: area scaled by 1344/756 vertically -> threshold 576 px^2
    assert specialist.is_small([0, 0, 24, 23], 1344, 756, "in32")  # 552
    assert not specialist.is_small([0, 0, 24, 24], 1344, 756, "in32")  # 576
    # in32 on 3840x2160: threshold 1024 * 3840 * 2160 / 1344^2 = 4702 px^2
    assert specialist.is_small([0, 0, 68, 69], 3840, 2160, "in32")  # 4692
    assert not specialist.is_small([0, 0, 69, 69], 3840, 2160, "in32")  # 4761
    with pytest.raises(ValueError):
        specialist.is_small([0, 0, 1, 1], 10, 10, "area")


def test_filter_and_score_scaling_leave_inputs_untouched():
    fb = [rec(1, 0, 0, 0, 10, 10, 0.8), rec(1, 0, 100, 100, 50, 50, 0.9)]
    small = specialist.eligible_fb(fb, 1344, 756, "coco32", 0.75)
    assert [r["bbox"] for r in small] == [[0, 0, 10, 10]]
    assert small[0]["score"] == pytest.approx(0.6)
    assert fb[0]["score"] == 0.8  # inputs not modified


def test_no_eligible_fb_returns_model_a_exactly():
    a = [
        rec(1, 0, 0, 0, 40, 40, 0.9),
        rec(1, 1, 0, 0, 40, 40, 0.5),
        rec(2, 3, 5.25, 6.5, 100, 80, 0.3),
    ]
    fb = [rec(1, 0, 200, 200, 90, 90, 0.99)]  # large: never eligible
    out = specialist.merge(a, fb, SIZES, "coco32", 1.0, 0.5)
    assert out == sorted(a, key=lambda r: (r["image_id"], -r["score"]))


def test_classwise_nms_and_tie_order():
    a = [rec(1, 0, 0, 0, 20, 20, 0.5)]
    fb = [
        rec(
            1, 0, 1, 1, 20, 20, 0.9
        ),  # same class, overlaps A: higher score wins, A suppressed
        rec(1, 2, 0, 0, 20, 20, 0.9),  # other class: kept
        rec(1, 0, 300, 300, 10, 10, 0.4),  # separate box: kept
    ]
    out = specialist.merge(a, fb, SIZES, "coco32", 1.0, 0.5)
    assert [(r["category_id"], r["bbox"][0]) for r in out] == [(0, 1), (2, 0), (0, 300)]
    # with multiplier 0.5, A (0.5) ties FB (0.45 < 0.5): A wins the overlap
    out = specialist.merge(a, fb, SIZES, "coco32", 0.5, 0.5)
    assert (0, 0) in [(r["category_id"], r["bbox"][0]) for r in out]
    assert (0, 1) not in [(r["category_id"], r["bbox"][0]) for r in out]
    # exact score tie: A comes first and suppresses the FB duplicate
    out = specialist.merge(a, [rec(1, 0, 1, 1, 20, 20, 0.5)], SIZES, "coco32", 1.0, 0.5)
    assert out == a


def test_global_cap():
    a = [rec(1, i % 8, 30 * i, 0, 20, 20, 0.5 + i / 1000) for i in range(498)]
    fb = [rec(1, 0, 30 * i, 500, 10, 10, 0.9) for i in range(5)]
    out = specialist.merge(a, fb, SIZES, "coco32", 1.0, 0.5, max_det=500)
    assert len(out) == 500
    assert sum(r["bbox"][1] == 500 for r in out) == 5  # all FB boxes outrank A
    kept_a = sorted(r["score"] for r in out if r["bbox"][1] == 0)
    assert kept_a[0] == pytest.approx(
        0.5 + 3 / 1000
    )  # the 3 lowest A boxes were capped away


def test_deterministic():
    a = [rec(i % 2 + 1, i % 8, i, i, 15, 15, (i * 37 % 101) / 101) for i in range(300)]
    fb = [
        rec(i % 2 + 1, i % 8, i + 2, i, 12, 12, (i * 53 % 97) / 97) for i in range(300)
    ]
    one = specialist.merge(a, fb, SIZES, "in32", 0.75, 0.6)
    two = specialist.merge(list(a), list(fb), dict(SIZES), "in32", 0.75, 0.6)
    assert json.dumps(one) == json.dumps(two)


@pytest.mark.parametrize(
    "bad",
    [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 1], "score": 0.5},
        {
            "image_id": 1,
            "category_id": 0,
            "bbox": [0, 0, 1, float("nan")],
            "score": 0.5,
        },
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, -1, 1], "score": 0.5},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 1, 1], "score": float("inf")},
        {"image_id": 1, "bbox": [0, 0, 1, 1], "score": 0.5},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 1, 1], "score": "x"},
    ],
)
def test_malformed_inputs_rejected(bad):
    with pytest.raises(ValueError):
        specialist.merge([rec(1, 0, 0, 0, 5, 5, 0.5)], [bad], SIZES, "coco32", 1.0, 0.5)


def test_invalid_parameters_and_missing_size():
    a = [rec(1, 0, 0, 0, 5, 5, 0.5)]
    with pytest.raises(ValueError):
        specialist.merge(a, [], SIZES, "coco32", 1.0, 0.4)  # IoU < 0.5
    with pytest.raises(ValueError):
        specialist.merge(a, [], SIZES, "coco32", 0.0, 0.5)
    with pytest.raises(ValueError):
        specialist.merge(a, [], SIZES, "nope", 1.0, 0.5)
    with pytest.raises(ValueError):
        specialist.merge([rec(9, 0, 0, 0, 5, 5, 0.5)], [], SIZES, "coco32", 1.0, 0.5)
