# CH-14b — authorized short RT-DETR-l extension of CH-14

Authorized by student Badabuba on 2026-11-10, after the competition deadline was extended. The scope was fixed by the student:
Stage A (evaluate the existing CH-14 epoch-1 checkpoint), then Stage B (one 2-epoch fine-tune of the fallback at 0.1x the
learning rate) **only if Stage A did not win**. No other search. `results/ch-14-final/` and the CH-14 negative result are unchanged.
Worker: Claude. No CSV was uploaded by Claude.

## Order of events (all on branch `ch-14b-rtdetr-short-finetune`)

1. `selection_rule.md` + `scripts/ch14b_infer.py` committed (6895821) before any CH-14b number existed.
2. `check_identity.py stageA` → `stageA_identity.json`: 8/8 checks pass (sha256 `6ff4a430…5b0b` = `results/ch-14-final/checkpoints.json`,
   RT-DETR-l with 300 queries and 8 classes, epoch field 0 of 6, imgsz 1344, batch 1, seed 42, fraction 1.0, data = the
   train_final export, run record train_final 6,072 frames, split file hash unchanged). The first run of this checker
   failed 2 checks because of two checker bugs (a wrongly typed class-name list; a CRLF checkout of the split file), which were
   fixed before any inference. The checkpoint was never in question.
3. A1, val_fast, CH-13 pipeline: **0.2892** ≥ gate 0.2725, so competitive (committed in a20306f before val_full ran).
   The single pass reproduces the CH-14 curve value 0.2790 exactly.
4. A2, val_full, once: **0.2860** vs the fallback 0.2696 → rule R passes (d1d25b0). Stage B therefore did **not** run (its config,
   tests and an init-check option were drafted but are not committed, since they were never used).
5. Same-session speed, test inference (3,811 frames, 48 resumable shards), CSV, both validators.

## val_full (2,846 frames, project evaluator, maxDets 500, CH-13 pipeline)

| model | mAP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---|---|---|---|---|---|---|
| F = rtdetr_baseline + CH-13 pipeline | 0.2696 | 0.5069 | 0.2453 | 0.1467 | 0.2375 | 0.4502 |
| A = CH-14 epoch0.pt + CH-13 pipeline | 0.2860 | 0.5358 | 0.2600 | 0.1504 | 0.2568 | 0.4812 |
| A, single pass (reported only) | 0.2762 | 0.5131 | 0.2541 | 0.1101 | 0.2462 | 0.4714 |

Rule R: lead +0.0163 ≥ 0.0063; 8 of 9 secondary checks won (lost: density ≥100, 0.3497 vs 0.3614); AP_S guard met; identity met.
Per-class APs: `per_class_stageA_epoch1.csv` (7 of 8 classes up; bicycle −0.0076). Density bins: `density_stageA_epoch1.csv`.
All evaluations in one table: `stages.csv`. Decision: `selection.json`.

Known limits: val_fast (714 frames) is a subset of val_full, and the epoch was chosen after the CH-14 curve was seen, so the val_full
lead is slightly optimistic. One run, no replicate. The R1 margin was not lowered.

## Speed and memory (RTX 5070 Laptop, same session, `speed/session.json`)

| model | single FPS | pipeline FPS | peak CUDA allocated |
|---|---|---|---|
| F rtdetr_baseline | 13.73 | 2.27 | 0.65 GiB |
| A epoch0.pt | 13.73 | 2.26 | 0.74 GiB |

Same architecture, so the same speed. In its own processes A peaks at the same 699,874,304 B (0.65 GiB) allocated as F
(val_full run and test run, `stageA_epoch1/*/run.json`). The 0.74 GiB above was measured for the second model loaded in the
speed process; its cause was not investigated. Full test inference took 1,729.5 s (2.20 FPS).

## Test candidate (not uploaded, not in Git)

`C:/UCU/AI/hw2/MOVD-ch14b/submissions/final_rtdetr_epoch1_ch13_fliptile_nms05_test.csv`, 172,680,339 bytes, sha256
`e587c68b517d6d59f90122946d8a45ef5beac4fb8ae3b24f66ced1268def0da7`, 3,811 rows, 1,875,374 detections (≤ 500 per row).
Project validator exit 0 (`submission/project_validator.txt`); independent stdlib validator exit 0, 0 failures, 0 rows differing
from the raw predictions, checkpoint hash matched (`submission/independent_validator.json`). Summary: `final_submission.json`.

## Reproduce (CPU, from stored files)

    python results/ch-14b-rtdetr-short-finetune/make_evidence.py

GPU commands (already run): `scripts/ch14b_infer.py val --split val_fast|val_full`, `speed`, `test`, with
`--weights <MOVD-ch14>/outputs/final_rtdetr/train/weights/epoch0.pt --name stageA_epoch1 --device 0`.

## Student TODOs (Claude does not write conclusions)

- TODO (students): decide whether to upload this CSV, and record the upload and its public score in `results/submissions.csv`.
- TODO (students): make and record the 2 Kaggle final selections (G2c, by internal validation).
- TODO (students): interpret why epoch 1 beats epoch 6 of the same run and the train_rubric baseline, and discuss the selection bias above.
