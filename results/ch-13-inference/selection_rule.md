# CH-13 search space and selection rule (written before any CH-13 result was produced)

Written 2026-11-09 about 13:55 Kyiv, after the claim (`04d757a`) and before the first CH-13 inference pass.
No CH-13 metric existed when this file was committed. Changing it after results exist would need a recorded
deviation in the evidence log.

## Fixed reference R

* Checkpoint: CH-11 `rtdetr_baseline` `last.pt` (MOVD-ch11 `outputs/rtdetr_baseline/train/weights/last.pt`,
  sha256 `da5d36258918e7fdddcb9cfe6e87f6a78ef8ee1a04e65deb3c0eedb42fb6e872`, the hash CH-12 recorded in
  `results/ch-12-comparison/speed_same_session/summary.json`). The same file is used for every row; no training,
  no change to the weights.
* Inference: Ultralytics 8.4.207 RT-DETR predictor, FP32, batch 1, imgsz 1344 (stretched to 1344x1344, as in
  training), one pass, decoder top-k 300 over (query, class) pairs (the Ultralytics default; 300 queries), conf
  > 0.001, no NMS (RT-DETR is query-based), no TTA, single model, then the shared top-500-per-image cap (not binding).
* CH-11 val_full numbers of R: mAP 0.2577, AP50 0.4775, AP_small 0.1296 (`results/rtdetr_baseline/metrics.json`).
  CH-13 re-derives R from cached raw decoder outputs and must reproduce these numbers (identity check).

## Splits

* Tuning (every comparison and every selection): `val_fast` (714 frames, `configs/splits/val_fast.json`).
  This follows the user's CH-13 instruction. It deviates from plan.md §4, where val_fast is listed for per-epoch
  curves only, and from the CH-13 scope line "on val_full"; both are recorded in the evidence log.
* `val_full` (2,846 frames): only R, the selected configuration S and each adopted single change. val_full
  numbers never change the selection, except the veto in rule 7. val_fast is a subset of val_full (every 4th
  frame), so val_full is not an independent hold-out; it is the protocol's headline split.
* The test images are not used in CH-13.

## Search space (each axis measured separately against R)

| Axis | Variants (R in bold) |
|---|---|
| 1 Inference input size | imgsz 1152, **1344**, 1536, 1728 (square, stretched like training) |
| 2 Tiling / slicing | R + a 2x2 grid of tiles of 0.6 W x 0.6 H (0.2 overlap in each dimension, same aspect as the frame, so each tile is stretched like a training frame, about 1.67x zoom), every tile at imgsz 1344; tile boxes shifted to frame pixels; merged with the full-frame pass by class-wise NMS at IoU 0.5, 0.6 or 0.7; then the 500 cap. 3 variants |
| 3a Confidence threshold | conf **0.001**, 0.01, 0.05 |
| 3b Decoder top-k (max_det, <= 500) | k 100, **300**, 500 (k 500 keeps more than one class hypothesis per query; still <= 500 per image) |
| 3c Post-hoc NMS | **none**, class-wise NMS IoU 0.7, 0.5 (RT-DETR has no NMS; this only tests whether duplicates exist) |
| 4a TTA: horizontal flip | R + the flipped 1344 pass (boxes un-flipped), merged by class-wise NMS 0.6 or by WBF 0.6. 2 variants |
| 4b TTA: two scales | R + the 1536 pass, merged by class-wise NMS 0.6 or WBF 0.6. 2 variants (fixed now, not chosen from axis 1) |
| 5 Ensemble (only if it gains) | R + CH-11 `rtdetr_q500` `last.pt` (500 queries, same budget, val_full 0.2389) at 1344, merged by WBF 0.6 or class-wise NMS 0.6. 2 variants |

WBF = weighted boxes fusion (score-weighted box average of each cluster, cluster score = mean over the passes,
counting a pass with no box as 0); implemented in `movd/infer.py`, no new dependency.

## Selection rule

1. Primary metric: val_fast mAP (AP@0.50:0.05:0.95, project evaluator `movd/evaluate.py`, maxDets 500).
2. A variant passes if its val_fast mAP >= R + 0.003 and its val_fast AP_small >= R - 0.003.
   0.003 is about half the CH-10 replicate spread of a training run (0.0063 mAP); inference is deterministic, so
   the remaining noise is the choice of 714 frames.
3. Speed constraint: the full pipeline must reach >= 1.6 FPS end to end on the RTX 5070 (same-session benchmark,
   first 30 val_full frames preloaded, includes every pass and the merge), so the 3,811 test frames take
   <= 40 min, the upper end of the CH-14 test-inference budget in plan.md (about 20-40 min).
4. Per axis, the passing variant with the highest val_fast mAP is the axis winner. No passing variant: the
   axis keeps R and is recorded as neutral or negative.
5. Combination: axis winners are added greedily in order of their individual val_fast gain. An addition is
   kept only if the combined val_fast mAP improves by >= 0.002 over the current stack and rule 3 still holds.
   When two winners act on the same pass (for example size and tiling), the combination uses the winner's
   setting on every pass.
6. An ensemble (axis 5) needs a larger margin, >= R + 0.005, because it doubles the cost and fixes a second
   member checkpoint; if adopted, its members are frozen literally.
7. Final check: R and S (and each adopted single change) on val_full. If S's val_full mAP < R's, R is frozen
   instead and both are reported.
8. Every variant is reported, including neutral and negative ones. Nothing is deleted.
