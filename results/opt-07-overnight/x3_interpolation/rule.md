# OPT-07 x3_interpolation — Conservative two-sided one-frame interpolation (frozen before evaluation)

Written and committed **before** this candidate was evaluated on any split. Not changed after results are seen.

## Hypothesis

A confident object seen in t-1 and t+1 but missing in t is a one-frame miss; adding the midpoint box at a
reduced score recovers it with limited false-positive risk.

## Exact deterministic rule

For every frame t with both t-1 and t+1 in the same clip, per class: keep boxes of score >= 0.3 in t-1 and t+1;
pair them one-to-one greedily by IoU (highest first, ties row-major), pairs with IoU >= 0.5. For each pair, the
midpoint box (mean of the two xyxy boxes) is added to t with score 0.5 x min(s_prev, s_next), unless t already has a
same-class box (any score) with IoU >= 0.5 with it. All additions are computed from the original predictions (added
boxes never seed other additions). Then each frame keeps its top 500 by score (stable). `movd.opt07.interpolation`.

Input: Model A cached CH-13-pipeline val_full predictions (sha256 `fccda2b3…036c`) and test predictions
(sha256 `5f9c4913…930e`), unchanged otherwise. Neighbours: frame number +-1 within the same clip (val: COCO `video` /
`frame_id`; test: `test_images.csv` `clip` / `frame`).

Frozen parameters (`movd/opt07.py` EXPERIMENTS, checked by `scripts/opt07_loop.py run`):

    {"exist_iou": 0.5, "factor": 0.5, "kind": "interpolation", "max_det": 500, "min_score": 0.3, "pair_iou": 0.5}

## Evaluation and rejection criteria

Evaluated **once** on val_full (no val_fast screening, no grid) with the project evaluator, against Model A on the
same subsets (`scripts/opt07_loop.py run x3_interpolation`). Rejected if any of P1–P7 in
[`../progress.md`](../progress.md) fails: P1 val_full mAP < A + 0.0005; P2 width <= 1920 subset mAP < A + 0.0005;
P3 AP_small loss > 0.003; P4 density >= 100 loss > 0.005; P5 fewer than 4 of 7 clips non-negative; P6 without-4K
mAP < A + 0.0005; P7 non-deterministic predictions. If promoted: one lossless test CSV, both validators, never uploaded.
