"""OPT-09 grids and the parameterised five-frame boost on tiny synthetic clips (CPU only)."""

from __future__ import annotations

import random

import pytest

from movd import opt07, opt08, opt09


def det(image_id, score, x=10.0, cat=1, w=20.0):
    return {
        "image_id": image_id,
        "category_id": cat,
        "bbox": [x, 10.0, w, 20.0],
        "score": score,
    }


def clip(n, clip_name="c"):
    return opt07.neighbour_table((i, clip_name, i) for i in range(n))


def random_clip(seed, frames=7, boxes=6):
    rng = random.Random(seed)
    preds = []
    for f in range(frames):
        for _ in range(boxes):
            preds.append(
                det(
                    f,
                    rng.random(),
                    x=rng.uniform(0, 40),
                    cat=rng.choice((1, 2)),
                    w=rng.uniform(10, 30),
                )
            )
    return preds, clip(frames)


@pytest.mark.parametrize("seed", range(5))
def test_e6_settings_reproduce_opt08_b3_exactly(seed):
    preds, nbrs = random_clip(seed)
    ours = opt09.five_frame_boost(preds, nbrs, opt09.E6["iou"], opt09.E6["strength"])
    ref = opt08.broader_support(preds, nbrs, "five_frame_mean")
    assert [p["score"] for p in ours] == [p["score"] for p in ref]


def test_every_candidate_changes_exactly_one_e6_component():
    assert sum(len(g) for g in opt09.GRIDS.values()) <= 3 + 5 + 4
    assert len(opt09.GRIDS["a1_border_margin"]) <= 3
    assert len(opt09.GRIDS["a2_persistence"]) <= 5
    assert len(opt09.GRIDS["b_soup_weight"]) <= 4
    for grid, cands in opt09.GRIDS.items():
        for cid in cands:
            assert len(opt09.changed_keys(opt09.candidate(grid, cid))) == 1


def test_grid_values_are_untested():
    margins = [c["margin"] for c in opt09.GRIDS["a1_border_margin"].values()]
    assert not set(margins) & set(opt09.TESTED["margin"])
    weights = [
        opt09.SOUP_WEIGHTS[c["soup"]] for c in opt09.GRIDS["b_soup_weight"].values()
    ]
    assert not set(weights) & set(opt09.TESTED["soup_weight_model_A"])
    for c in opt09.GRIDS["a2_persistence"].values():
        assert (
            c.get("iou", 0.5) not in opt09.TESTED["persistence_iou"] or "iou" not in c
        )
        assert (
            c.get("strength", 1.0) not in opt09.TESTED["persistence_strength"]
            or "strength" not in c
        )


def test_strength_scales_the_boost_and_never_lowers():
    preds = [det(0, 0.9), det(1, 0.3), det(2, 0.9)]
    nbrs = clip(3)
    full = opt09.five_frame_boost(preds, nbrs, strength=1.0)[1]["score"]
    half = opt09.five_frame_boost(preds, nbrs, strength=0.5)[1]["score"]
    strong = opt09.five_frame_boost(preds, nbrs, strength=1.5)[1]["score"]
    assert full == pytest.approx(0.7)
    assert half == pytest.approx(0.5)
    assert strong == pytest.approx(0.9)
    capped = opt09.five_frame_boost(
        [det(0, 1.0), det(1, 0.6), det(2, 1.0)], nbrs, strength=3.0
    )
    assert capped[1]["score"] == 1.0
    # boxes above their mean are unchanged; ends of the clip are unchanged
    out = opt09.five_frame_boost(preds, nbrs, strength=1.5)
    assert out[0]["score"] == 0.9 and out[2]["score"] == 0.9


def test_iou_threshold_controls_support():
    # middle box shifted: IoU with neighbours = 15/25 = 0.6
    preds = [det(0, 0.9), det(1, 0.3, x=15.0), det(2, 0.9)]
    nbrs = clip(3)
    assert opt09.five_frame_boost(preds, nbrs, iou=0.55)[1]["score"] > 0.3
    assert opt09.five_frame_boost(preds, nbrs, iou=0.65)[1]["score"] == 0.3


def test_boxes_are_never_moved_added_or_removed():
    preds, nbrs = random_clip(11)
    out = opt09.five_frame_boost(preds, nbrs, iou=0.45, strength=1.25)
    assert len(out) == len(preds)
    for a, b in zip(preds, out, strict=True):
        assert (a["image_id"], a["category_id"], a["bbox"]) == (
            b["image_id"],
            b["category_id"],
            b["bbox"],
        )
        assert b["score"] >= a["score"]
