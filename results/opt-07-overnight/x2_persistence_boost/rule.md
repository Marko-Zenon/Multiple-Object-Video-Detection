# OPT-07 x2_persistence_boost — Temporal persistence boost (frozen before evaluation)

Written and committed **before** this candidate was evaluated on any split. Not changed after results are seen.

## Hypothesis

A box that is matched in both adjacent frames is more often a true object; raising its score to the 3-frame
mean when that mean is higher promotes temporally consistent low-score detections.

## Exact deterministic rule

For every Model A box b in frame t with both t-1 and t+1 in the same clip: take the same-class box with the highest
IoU with b in t-1 and in t+1 (first maximum on ties). If both IoUs are >= 0.5, score(b) := max(score(b), (s_prev +
score(b) + s_next) / 3) (original scores). Otherwise unchanged. No box added or removed. `movd.opt07.persistence_boost`.

Input: Model A cached CH-13-pipeline val_full predictions (sha256 `fccda2b3…036c`) and test predictions
(sha256 `5f9c4913…930e`), unchanged otherwise. Neighbours: frame number +-1 within the same clip (val: COCO `video` /
`frame_id`; test: `test_images.csv` `clip` / `frame`).

Frozen parameters (`movd/opt07.py` EXPERIMENTS, checked by `scripts/opt07_loop.py run`):

    {"iou": 0.5, "kind": "persistence_boost"}

## Evaluation and rejection criteria

Evaluated **once** on val_full (no val_fast screening, no grid) with the project evaluator, against Model A on the
same subsets (`scripts/opt07_loop.py run x2_persistence_boost`). Rejected if any of P1–P7 in
[`../progress.md`](../progress.md) fails: P1 val_full mAP < A + 0.0005; P2 width <= 1920 subset mAP < A + 0.0005;
P3 AP_small loss > 0.003; P4 density >= 100 loss > 0.005; P5 fewer than 4 of 7 clips non-negative; P6 without-4K
mAP < A + 0.0005; P7 non-deterministic predictions. If promoted: one lossless test CSV, both validators, never uploaded.
