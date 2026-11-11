# OPT-08 e2_tracklet_smoothing: E2 — short-tracklet confidence smoothing (6 frozen candidates)

Written and committed **before** any candidate of this experiment was evaluated on any split. Not changed
after results are seen.

## Hypothesis

x2 uses one frame on each side. A short tracklet gives a longer, less noisy estimate of how confident the model is in
an object; a score aggregated over it may rank true objects above clutter better than the 3-frame x2 rule.

## Exact deterministic rule (`movd.opt08.tracklet_smoothing`)

Input: Model A cached CH-13-pipeline predictions (val_full `fccda2b3…036c`, test `5f9c4913…930e`); not x2's output.
Links: box i in frame t and box j in frame t+1 of the same clip and class are linked when j is i's best IoU match,
i is j's best IoU match (first maximum on ties; the x2 matching made one-to-one) and IoU >= 0.5. Linked boxes form
chains (tracklets). Only boxes with a tracklet member on both sides are changed (as x2). Window = the tracklet
members within +-k positions (truncated at the tracklet ends). Aggregations, from original scores only:
`boost_mean` max(s, mean), `replace_mean` mean (can lower), `boost_median` max(s, median), `boost_triangular`
max(s, weighted mean with weights k+1-|offset|). Boxes are never moved, added or removed.

Frozen parameters (`movd/opt08.py` EXPERIMENTS, checked by `scripts/opt08_loop.py screen`):

    t1 {"agg": "boost_mean", "k": 2, "kind": "tracklet"}
    t2 {"agg": "boost_mean", "k": 3, "kind": "tracklet"}
    t3 {"agg": "boost_mean", "k": 5, "kind": "tracklet"}
    t4 {"agg": "replace_mean", "k": 2, "kind": "tracklet"}
    t5 {"agg": "boost_median", "k": 2, "kind": "tracklet"}
    t6 {"agg": "boost_triangular", "k": 3, "kind": "tracklet"}

Comparison baseline: OPT-07 x2 (val_full predictions `6c3a929f…2c03`, public 0.2805168313).

## Stages and gate

Fixed in [`../TASK.md`](../TASK.md) before this rule: S1 discovery screen of every listed candidate (4 clips only; the
selection is the highest discovery mAP, ties to the earlier candidate; S1 passes if its delta vs x2 >= +0.0003),
S2 the selection once on the 3 confirmation clips (delta vs x2 >= 0), S3 val_full once only after S1 and S2 pass,
judged by U1-U8 against x2. Rejected if any stage or gate check fails; never retuned after a result.
`python scripts/opt08_loop.py screen|confirm|full|determinism|test <experiment>`.
