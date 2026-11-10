# CH-14b selection rule (committed before any CH-14b result)

CH-14b is a limited extension of CH-14, authorized by student Badabuba on 2026-11-10 after the competition deadline
was extended. The scope is fixed by the student: Stage A, then Stage B only if Stage A does not win. Nothing else is
searched (no other model, learning rate, resolution, epoch count, ensemble or tracking; no tuning on val_full).
The original CH-14 rule (`results/ch-14-final/selection_rule.md`) and its result (`results/ch-14-final/selection.json`,
fallback kept) stay unchanged. No CSV is uploaded by Claude, and the Kaggle selections stay with the students.

When this file was committed, the only CH-14b-relevant numbers that existed were the CH-14 ones already on `develop`:
the single-pass val_fast curve of `final_rtdetr` (epoch 1: 0.2790, epoch 6: 0.2381, `results/ch-14-final/epochs.csv`)
and the fallback evidence below. No CH-13-pipeline number of any CH-14 epoch checkpoint other than `last.pt` existed.

## Fallback F (unchanged)

CH-11 `rtdetr_baseline` `last.pt`, sha256 `da5d36258918e7fdddcb9cfe6e87f6a78ef8ee1a04e65deb3c0eedb42fb6e872`, with
the CH-13 pipeline `configs/runs/final_inference.json`.
val_fast: mAP 0.2725 (`final_inference.json` `evidence.val_fast.selected`).
val_full: mAP 0.2696, AP50 0.5069, AP75 0.2453, AP_S 0.1467, AP_M 0.2375, AP_L 0.4502
(`results/ch-13-inference/runs/val_full/combo_flip_tile_nms05/metrics.json`).
Its test CSV (sha256 `73f73e21…793b`) scored 0.2328849249 on the public LB (student-reported).

All evaluations below use the project evaluator (`movd/evaluate.py`, maxDets 500) and the CH-13 pipeline exactly as
frozen in `configs/runs/final_inference.json` (run by `scripts/ch14_infer.py`, the CH-14 code, through the thin
wrapper `scripts/ch14b_infer.py` that only changes the output folders). The single-pass numbers come from the same
forward pass and are reported, but never decide.

## Rule R (the CH-14 rule B, thresholds unchanged)

A candidate X replaces F as the primary submission only if **all** of these hold on val_full (CH-13 pipeline):

1. R1: mAP(X) ≥ mAP(F) + **0.0063** (the measured run-to-run spread, CH-10 replicate, CH-12 rule 3).
2. R2: X leads F on **at least 5 of the 9 secondary checks**: AP50, AP75, AP_S, AP_M, AP_L and the 4 density bins
   (same bin edges as `results/ch-12-comparison/density.csv`). Every check X loses is reported.
3. R3: AP_S(X) ≥ AP_S(F) − 0.003 (the CH-13 small-object guard).
4. R4: the identity checks of the candidate (below) pass.

A lead inside the spread is a tie, not a win; on a tie or any failed criterion F is kept.

**The one recorded deviation from CH-14 rule B:** CH-14 B4 required "6 completed epochs", which no epoch-1 or
2-epoch checkpoint can meet. R4 replaces it with the identity checks of each stage below (exact checkpoint hash,
train_final, non-smoke, the stated schedule). R1–R3 are not weakened. Known bias, reported with the result:
val_fast (714 frames) is a subset of val_full (2,846 frames), so a checkpoint picked on val_fast has a small
optimistic bias on val_full. The R1 margin is not lowered to compensate for anything.

## Stage A: the existing CH-14 epoch-1 checkpoint

Candidate A = `outputs/final_rtdetr/train/weights/epoch0.pt` of the CH-14 run (saved after epoch 1 of 6, in the
3-epoch warmup), sha256 `6ff4a430188405b5d13356872c63f141c4d3dbc7b747983cdf733ce386aa5b0b`
(`results/ch-14-final/checkpoints.json`).

A0 (identity, R4 for A), checked before any inference:
* the file hash equals the value above;
* RT-DETR-l: `RTDETRDecoder` head, 300 queries, 8 classes (person … motor), same layer count as `rtdetr-l.pt`;
* checkpoint `epoch` field 0; `train_args`: epochs 6, imgsz 1344, batch 1, seed 42, fraction 1.0 (not a smoke), data
  = the `train_final` export;
* the CH-14 run record `results/final_rtdetr/config.json`: train_split `train_final`, export manifest 6,072 frames,
  split file sha256 equal to `configs/splits/train_final.json` today.

A1 (gate, val_fast, CH-13 pipeline): A is "competitive" only if mAP_val_fast(A) ≥ **0.2725** (F on val_fast, same
pipeline). If not, A is not evaluated on val_full and Stage B runs.

A2: if A1 passes, A is evaluated **once** on val_full and compared with F by rule R.
If A passes R, A is selected and Stage B is skipped. Otherwise Stage B runs.

## Stage B: one short fine-tune from F (only if Stage A does not win)

Fresh fine-tune, not an optimizer resume, config `configs/runs/ch14b_finetune.json` (committed with its tests before
the run):
* RT-DETR-l, initialized from F (`rtdetr_baseline` `last.pt`, sha256 `da5d3625…e872`), fresh optimizer;
* train_final (6,072 frames, manifest checked by `scripts/train_ultra.py`), imgsz 1344, batch 1, AMP, seed 42,
  frozen ignore masking (the CH-14 masked export), flip-only augmentation;
* at most **2 epochs**;
* lr0 = 0.1 × the effective initial learning rate of `rtdetr_baseline`. That run used `optimizer=auto`, which in
  Ultralytics 8.4.207 picks AdamW with lr0 = round(0.002 · 5 / (4 + nc), 6) = **0.000833** for nc = 8, beta1 0.9,
  and warmup_bias_lr 0.0. So lr0 = **0.0000833**, with optimizer AdamW, momentum 0.9 and warmup_bias_lr 0.0 set
  explicitly (the CH-11 `rtdetr_lr1e4_wb0` method: setting the optimizer explicitly is the only way for lr0 not to be
  ignored, and the other two values reproduce what auto picks). Every other training setting is unchanged
  (weight decay 0.0005, nbs 64, lrf 0.01, linear schedule, warmup_epochs 3; Ultralytics 8.4.207 caps warmup at
  epochs − 1 = 1 epoch for a 2-epoch run, so epoch 1 warms up and epoch 2 runs at lr0 · lf(1)).

B-epoch choice: both epoch checkpoints (`epoch0.pt`, `epoch1.pt`) are evaluated on **val_fast only** with the CH-13
pipeline; the higher pipeline mAP wins (an exact tie picks epoch 2, the end of the schedule). Only the chosen
checkpoint is evaluated on val_full, once, and compared with F by rule R.

R4 for B: init hash `da5d3625…e872` recorded in the run config and every init tensor transferred unchanged; train_final
6,072 frames; not a smoke; epochs 2; lr0 0.0000833; the chosen checkpoint's hash recorded.

If B passes R, B is selected; otherwise F stays the primary submission and the negative result is kept.

## Submission (only if A or B wins)

Test inference over all 3,811 frames with the CH-13 pipeline (resumable per-clip shards), the lossless CSV from the
CH-05 tooling (`scripts/make_submission.py`), then the project validator and the independent stdlib validator. The CSV,
shards and checkpoints are not committed. If nothing wins, the existing fallback CSV is kept and not regenerated.
