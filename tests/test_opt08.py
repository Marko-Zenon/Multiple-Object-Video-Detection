"""OPT-08 transforms on tiny synthetic clips (CPU only)."""

from __future__ import annotations

import pytest

from movd import opt07, opt08


def det(image_id, score, x=10.0, cat=1, w=20.0):
    return {
        "image_id": image_id,
        "category_id": cat,
        "bbox": [x, 10.0, w, 20.0],
        "score": score,
    }


def clip(n, clip_name="c"):
    return opt07.neighbour_table((i, clip_name, i) for i in range(n))


def scores(preds):
    return [round(p["score"], 6) for p in preds]


def test_split_is_deterministic_and_order_free():
    clips = [f"uav{i:07d}_v" for i in range(7)]
    a = opt08.split_clips(clips)
    b = opt08.split_clips(list(reversed(clips)))
    assert a == b
    assert len(a["discovery"]) == 4 and len(a["confirmation"]) == 3
    assert sorted(a["discovery"] + a["confirmation"]) == sorted(clips)


def test_experiment_grids_are_small_and_frozen():
    sizes = {k: len(v) for k, v in opt08.EXPERIMENTS.items()}
    assert sizes["e1_x4_then_x2"] == 1
    assert sizes["e2_tracklet_smoothing"] <= 6
    assert sizes["e3_broader_support"] <= 4


def test_mutual_links_are_one_to_one():
    # frame 0 has two overlapping boxes, frame 1 one: only the better one links
    preds = [det(0, 0.9, x=10), det(0, 0.8, x=14), det(1, 0.7, x=10)]
    t = opt07.Table(preds)
    nxt = opt08.mutual_links(t, clip(2))
    assert list(nxt) == [2, -1, -1]


def test_tracklet_boost_mean_raises_only_interior_and_keeps_boxes():
    preds = [det(i, s) for i, s in enumerate([0.9, 0.9, 0.1, 0.9, 0.9])]
    out = opt08.tracklet_smoothing(preds, clip(5), "boost_mean", 2)
    assert [p["bbox"] for p in out] == [p["bbox"] for p in preds]
    assert scores(out)[0] == 0.9 and scores(out)[4] == 0.9  # endpoints unchanged
    assert scores(out)[2] == round((0.9 * 4 + 0.1) / 5, 6)
    assert all(o >= p["score"] for o, p in zip(scores(out), preds, strict=True))
    assert preds[2]["score"] == 0.1  # input not modified


def test_tracklet_replace_mean_can_lower():
    preds = [det(i, s) for i, s in enumerate([0.1, 0.9, 0.1])]
    out = opt08.tracklet_smoothing(preds, clip(3), "replace_mean", 2)
    assert scores(out)[1] == round(1.1 / 3, 6)


def test_tracklet_k1_boost_equals_x2_on_a_clean_track():
    preds = [det(i, s) for i, s in enumerate([0.5, 0.2, 0.8, 0.3])]
    a = opt08.tracklet_smoothing(preds, clip(4), "boost_mean", 1)
    b = opt07.persistence_boost(preds, clip(4), 0.5)
    assert scores(a) == scores(b)


def test_tracklet_classes_do_not_mix():
    preds = [det(0, 0.9), det(1, 0.1, cat=2), det(2, 0.9)]
    out = opt08.tracklet_smoothing(preds, clip(3), "boost_mean", 2)
    assert scores(out) == scores(preds)


def test_support_gap_tolerant_bridges_a_missed_frame():
    # box missing at t-1 (frame 1) but present at t-2: x2 does nothing, gap_tolerant boosts frame 2
    preds = [det(0, 0.9), det(2, 0.3), det(3, 0.9)]
    x2 = opt07.persistence_boost(preds, clip(4), 0.5)
    gap = opt08.broader_support(preds, clip(4), "gap_tolerant")
    assert scores(x2)[1] == 0.3
    assert scores(gap)[1] == round((0.9 + 0.3 + 0.9) / 3, 6)


def test_support_endpoint_one_sided_at_clip_start():
    preds = [det(0, 0.3), det(1, 0.9), det(2, 0.9)]
    out = opt08.broader_support(preds, clip(3), "endpoint_one_sided")
    assert scores(out)[0] == round(2.1 / 3, 6)
    # interior box: identical to x2
    x2 = opt07.persistence_boost(preds, clip(3), 0.5)
    assert scores(out)[1:] == scores(x2)[1:]


def test_support_five_frame_mean_needs_x2_condition():
    preds = [det(i, s) for i, s in enumerate([0.9, 0.9, 0.2, 0.9, 0.9])]
    out = opt08.broader_support(preds, clip(5), "five_frame_mean")
    assert scores(out)[2] == round((0.9 * 4 + 0.2) / 5, 6)
    lonely = [det(0, 0.9), det(2, 0.2), det(4, 0.9)]
    assert scores(opt08.broader_support(lonely, clip(5), "five_frame_mean")) == scores(
        lonely
    )


@pytest.mark.parametrize(
    "mode",
    ["endpoint_one_sided", "gap_tolerant", "five_frame_mean", "gap_tolerant_endpoint"],
)
def test_support_never_lowers_or_moves(mode):
    preds = [det(i, s, x=10 + i) for i, s in enumerate([0.4, 0.8, 0.2, 0.6, 0.1, 0.7])]
    out = opt08.broader_support(preds, clip(6), mode)
    assert [p["bbox"] for p in out] == [p["bbox"] for p in preds]
    assert all(o["score"] >= p["score"] for o, p in zip(out, preds, strict=True))


def test_apply_e1_score_step_is_x2():
    preds = [det(i, s) for i, s in enumerate([0.5, 0.2, 0.8])]
    params = opt08.EXPERIMENTS["e1_x4_then_x2"]["c1"]
    assert scores(opt08.apply(params, preds, clip(3))) == scores(
        opt07.persistence_boost(preds, clip(3), 0.5)
    )
