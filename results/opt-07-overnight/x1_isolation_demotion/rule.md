# OPT-07 x1_isolation_demotion — Temporal isolation demotion (frozen before evaluation)

Written and committed **before** this candidate was evaluated on any split. Not changed after results are seen.

## Hypothesis

A confident box that has no comparable same-class box near it in the previous or next frame is more often a
false positive (a one-frame flicker) than a box that persists; halving its score moves it below persistent boxes in
the per-class ranking without removing any box.

## Exact deterministic rule

For every Model A box b (class c, frame t of clip v): b is *supported* if frame t-1 or t+1 of the same clip
contains a class-c box with IoU(b, .) >= 0.3 and score >= 0.5 x score(b) (original scores). If b is not supported
and at least one adjacent frame exists, score(b) := 0.5 x score(b). Otherwise unchanged. No box is added or removed,
boxes keep their coordinates; every frame keeps <= 500 boxes. `movd.opt07.isolation_demotion`.

Input: Model A cached CH-13-pipeline val_full predictions (sha256 `fccda2b3…036c`) and test predictions
(sha256 `5f9c4913…930e`), unchanged otherwise. Neighbours: frame number +-1 within the same clip (val: COCO `video` /
`frame_id`; test: `test_images.csv` `clip` / `frame`).

Frozen parameters (`movd/opt07.py` EXPERIMENTS, checked by `scripts/opt07_loop.py run`):

    {"factor": 0.5, "iou": 0.3, "kind": "isolation_demotion", "rel_score": 0.5}

## Evaluation and rejection criteria

Evaluated **once** on val_full (no val_fast screening, no grid) with the project evaluator, against Model A on the
same subsets (`scripts/opt07_loop.py run x1_isolation_demotion`). Rejected if any of P1–P7 in
[`../progress.md`](../progress.md) fails: P1 val_full mAP < A + 0.0005; P2 width <= 1920 subset mAP < A + 0.0005;
P3 AP_small loss > 0.003; P4 density >= 100 loss > 0.005; P5 fewer than 4 of 7 clips non-negative; P6 without-4K
mAP < A + 0.0005; P7 non-deterministic predictions. If promoted: one lossless test CSV, both validators, never uploaded.
