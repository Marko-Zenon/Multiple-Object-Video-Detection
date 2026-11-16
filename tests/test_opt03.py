"""OPT-03 diagnostic tools: the traced merge equals the CH-13 merge, and the matcher labels boxes correctly."""

import json

import numpy as np

from movd import data, infer, opt03


def random_raws(seed=0, views=6, q=300, w=1000, h=600):
    rng = np.random.default_rng(seed)
    raws = []
    for _ in range(views):
        xy = rng.uniform(0, [w - 60, h - 60], size=(q, 2))
        wh = rng.uniform(4, 60, size=(q, 2))
        boxes = np.concatenate([xy, xy + wh], 1).astype(np.float32)
        raws.append(
            infer.Raw(boxes, rng.uniform(0, 1, size=(q, 8)).astype(np.float32) ** 3)
        )
    return raws


def test_traced_merge_equals_ch13_combine():
    config = json.loads(
        (data.REPO_ROOT / "configs" / "runs" / "final_inference.json").read_text(
            encoding="utf-8"
        )
    )["config"]
    for seed in range(3):
        raws = random_raws(seed)
        ref = infer.combine(raws, config)
        got = opt03.combine_traced(raws).dets(500)
        assert np.array_equal(ref.boxes, got.boxes)
        assert np.array_equal(ref.scores, got.scores)
        assert np.array_equal(ref.classes, got.classes)


def test_traced_sources_and_view_subset():
    raws = random_raws(1)
    t = opt03.combine_traced(raws, keep_views=[0])
    assert set(np.unique(t.src)) <= {0}
    single = infer.nms(infer.decode(raws[0]), 0.5)
    assert len(t.scores) == len(single)
    assert np.all(np.diff(t.scores) <= 0)


def test_match_frame_outcomes():
    gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float)
    gc = np.array([0, 2])
    crowd = np.array([[100, 100, 200, 200]], float)
    boxes = np.array(
        [
            [0, 0, 10, 10],  # TP person
            [0, 0, 10, 10],  # duplicate person
            [50, 50, 60, 60],  # wrong class (person on a car)
            [120, 120, 130, 130],  # inside crowd -> ignored
            [300, 300, 310, 310],  # background
        ],
        float,
    )
    out, hit = opt03.match_frame(boxes, np.array([0, 0, 0, 0, 0]), gt, gc, crowd)
    assert out.tolist() == [
        opt03.TP,
        opt03.FP_DUP,
        opt03.FP_CLASS,
        opt03.IGNORED,
        opt03.FP_BG,
    ]
    assert hit.tolist() == [0, -1]


def test_inner_tile_edge():
    window = (0, 0, 600, 360)  # top-left tile of a 1000 x 600 frame
    boxes = np.array(
        [[0, 0, 20, 20], [590, 10, 599, 30], [10, 340, 30, 359], [100, 100, 120, 120]],
        float,
    )
    assert opt03.touches_inner_tile_edge(boxes, window, 1000, 600).tolist() == [
        False,
        True,
        True,
        False,
    ]


def test_edge_filter_identity_and_effect():
    config = json.loads(
        (data.REPO_ROOT / "configs" / "runs" / "final_inference.json").read_text(
            encoding="utf-8"
        )
    )["config"]
    raws = random_raws(2)
    ref = infer.combine(raws, config)
    same = opt03.edge_filter_combine(raws, 1000, 600, 1.0, 0.01, config)
    assert np.array_equal(ref.boxes, same.boxes) and np.array_equal(
        ref.scores, same.scores
    )


def test_edge_filter_drops_only_tile_border_boxes():
    config = json.loads(
        (data.REPO_ROOT / "configs" / "runs" / "final_inference.json").read_text(
            encoding="utf-8"
        )
    )["config"]
    # Tile 0 of a 1000 x 600 frame is (0, 0, 600, 360): its right edge x = 600 is an inner edge.
    border = [550.0, 100.0, 600.0, 150.0]
    inside = [100.0, 100.0, 150.0, 150.0]
    empty = infer.Raw(np.zeros((1, 4), np.float32), np.zeros((1, 8), np.float32))

    def one(box, cls, score):
        s = np.zeros((1, 8), np.float32)
        s[0, cls] = score
        return infer.Raw(np.array([box], np.float32), s)

    # full view: the border box as class 2; tile0: the same box as class 0 and an inside box as class 1.
    tile0 = infer.Raw(
        np.array([border, inside], np.float32), np.zeros((2, 8), np.float32)
    )
    tile0.scores[0, 0], tile0.scores[1, 1] = 0.9, 0.8
    raws = [one(border, 2, 0.7), empty, tile0, empty, empty, empty]
    kept = opt03.edge_filter_combine(raws, 1000, 600, 0.0, 0.01, config)
    assert sorted(kept.classes.tolist()) == [1, 2]  # tile border box (class 0) dropped
    half = opt03.edge_filter_combine(raws, 1000, 600, 0.5, 0.01, config)
    assert np.isclose(
        half.scores[half.classes == 0][0], 0.45
    )  # down-weighted, not dropped
    assert np.isclose(half.scores[half.classes == 2][0], 0.7)  # full view untouched


def test_edge_margins_frame_edges_are_open():
    t = opt03.edge_margins((0, 0, 600, 360), 1000, 600, 0.01)
    assert (
        t[0] == -np.inf and t[1] == -np.inf and t[2] == 594 and np.isclose(t[3], 356.4)
    )
    boxes = np.array([[0, 0, 10, 10], [590, 0, 595, 10], [0, 350, 10, 357]], float)
    assert opt03.border_mask(boxes, t).tolist() == [False, True, True]
