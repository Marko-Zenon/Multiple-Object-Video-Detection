# OPT-02 — predeclared candidate grid and selection rule

Written and committed **before any OPT-02 metric was produced** (2026-11-10, Badabuba / worker Claude).
The grid, the gates and the tie-breaks are fixed by the student's request and are not changed after results are seen.

## Models (identities checked against committed evidence before use)

| | Model A (current best) | Model B (previous fallback) |
|---|---|---|
| Checkpoint | CH-14 `final_rtdetr` `epoch0.pt` (epoch 1 of 6, train_final 6,072 frames, imgsz 1344) | CH-11 `rtdetr_baseline` `last.pt` (train_rubric 2,800 frames x 6 epochs, imgsz 1344) |
| Checkpoint sha256 | `6ff4a430188405b5d13356872c63f141c4d3dbc7b747983cdf733ce386aa5b0b` | `da5d36258918e7fdddcb9cfe6e87f6a78ef8ee1a04e65deb3c0eedb42fb6e872` |
| Inference | frozen CH-13 `configs/runs/final_inference.json` | same |
| Registered val_full mAP | 0.2860 (`results/ch-14b-rtdetr-short-finetune/`) | 0.2696 (`results/ch-13-inference/`) |
| Kaggle public | 0.2774301362 (user-reported, rank 37) | 0.2328849249 (user-reported) |
| Test CSV sha256 | `e587c68b517d6d59f90122946d8a45ef5beac4fb8ae3b24f66ced1268def0da7` | `73f73e219dfbf3b674277703f7314d14c316d0800bcd046634aaecd27f2b793b` |

## Inputs and their limitation (stated before testing)

The cached predictions of both models are the **final CH-13 pipeline outputs, after the per-frame top-500 cap**.
No pre-cap (post-NMS, uncapped) predictions were ever saved for Model A, nor for the Model B test set
(`scripts/ch14_infer.py` and the CH-13 test run store only `combine()` output, which already applies the cap).
3,695 of 3,811 Model B test frames are at the 500 cap, so the cap is active on most frames.

Consequence: the ensemble merges two capped lists (up to 1,000 boxes per frame) and caps again at 500. A detection that
would rank below 500 inside its own model's list cannot enter the ensemble, even if it would survive the cross-model NMS.
This only removes low-score boxes, but it is not identical to merging the uncapped lists. The same inputs are used on
val and on test, so the val measurement is exactly the procedure that would produce the test CSV. Fresh GPU inference to
recover pre-cap lists is **not** done (the student's instruction: no GPU unless genuinely required; the experiment is
answerable on the cached files).

| Input | File (outside Git) | sha256 (committed evidence) |
|---|---|---|
| A val_fast | `MOVD-ch14b/outputs/ch14b/stageA_epoch1/val_fast/pipeline_predictions.json` | `87137d7d…275a0e` (CH-14b run.json) |
| A val_full | `MOVD-ch14b/outputs/ch14b/stageA_epoch1/val_full/pipeline_predictions.json` | `fccda2b3…7036c` (CH-14b run.json) |
| A test | `MOVD-ch14b/outputs/ch14b/stageA_epoch1/test/test_predictions.json` | `5f9c4913…87930e` (CH-14b final_submission.json) |
| B val_fast | `MOVD-ch13/outputs/ch13/preds/val_fast_combo_flip_tile_nms05.json` | no committed hash: identity = re-evaluation must reproduce `results/ch-13-inference/runs/val_fast/combo_flip_tile_nms05/metrics.json` exactly |
| B val_full | `MOVD-ch13/outputs/ch13/preds/val_full_combo_flip_tile_nms05.json` | same check against `runs/val_full/combo_flip_tile_nms05/metrics.json` |
| B test | `MOVD-ch13/outputs/ch13_test/test_predictions.json` | `184ddccc…647ccd` (CH-14 final_submission.json) |

Model A's re-evaluated cached predictions must also reproduce its registered metrics exactly (val_fast 0.2892, val_full 0.2860).

## Merge (identical for every candidate)

1. Take Model A's cached detections with score x `w_A` and Model B's with score x `w_B` (for the specialist: A without
   bicycle, B only bicycle, both weights 1.0).
2. Union per frame (A first, then B, each in cached order), stable sort by descending weighted score.
3. Class-wise NMS (`movd.infer.nms`, torchvision `batched_nms`) at IoU `t`.
4. Global top 500 per frame by weighted score (`movd.infer.cap`, stable).
5. Output records keep the cached `bbox` values unchanged (no xywh/xyxy round trip); only the score is the weighted one.

No WBF, Soft-NMS, tracking or other model.

## Candidate grid (all seven, val_fast only)

| id | w_A | w_B | NMS IoU | routing |
|---|---:|---:|---:|---|
| c1_w100_nms05 | 1.0 | 1.0 | 0.5 | all classes from both |
| c2_w100_nms06 | 1.0 | 1.0 | 0.6 | all classes from both |
| c3_w075_nms05 | 1.0 | 0.75 | 0.5 | all classes from both |
| c4_w075_nms06 | 1.0 | 0.75 | 0.6 | all classes from both |
| c5_w050_nms05 | 1.0 | 0.5 | 0.5 | all classes from both |
| c6_w050_nms06 | 1.0 | 0.5 | 0.6 | all classes from both |
| c7_specialist_bicycle | 1.0 | 1.0 | 0.5 | bicycle from B only, the other 7 classes from A only |

Baseline: Model A with the frozen CH-13 inference (its cached val_fast predictions, unchanged).

## Gate on val_fast (all must hold; unrounded evaluator values)

- G-a: `mAP(cand) >= mAP(A) + 0.003`
- G-b: `AP_small(cand) >= AP_small(A) - 0.003`
- G-c: density bin `>=100` mAP: `mAP_bin(cand) >= mAP_bin(A) - 0.005`
- G-d: projected test runtime (fresh inference of every model the candidate needs, measured CH-14b / CH-13 test
  inference seconds, plus the measured merge time scaled to 3,811 frames) `<= 90 min`

## Choice among passing candidates (exactly one)

Highest val_fast mAP; tie -> higher AP_small; tie -> higher `>=100` bin mAP; tie -> lower projected runtime; tie -> lower id.
If **no** candidate passes: stop model selection, keep Model A and its 0.2774301362 submission, record OPT-02 as a negative result.

## val_full confirmation (the chosen candidate only, evaluated once)

The candidate replaces Model A only if, on val_full: `mAP >= mAP(A) + 0.003`, `AP_small >= AP_small(A) - 0.003` and
`>=100` bin `mAP >= mAP_bin(A) - 0.005` (A = Model A's cached val_full predictions, registered 0.2860).
Otherwise Model A stays and no CSV is generated.

## Test CSV (only if val_full passes)

Merge all 3,811 test frames with the frozen chosen settings from the cached test predictions, write the CSV with the CH-05
writer (`movd.submission`, lossless), run `scripts/validate_submission.py` and `scripts/check_submission_independent.py`.
Not uploaded and not committed by Claude.
