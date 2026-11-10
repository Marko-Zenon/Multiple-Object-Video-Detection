# CH-14 selection rule (committed 2026-11-09 23:12 Kyiv in 4d69460, before the CH-14 smoke or training started)

No CH-14 metric existed when this file was committed. Changing it after results exist needs a recorded deviation
in the CH-14 evidence log. `plan.md` (CH-14 scope 2 and 5, §7 fallback) and `configs/runs/final.json`
(`fallback_and_second_selection`) say *what* is compared, but not which epoch checkpoint is used or what margin
counts as "better". This rule fills those two gaps with the conventions CH-11 and CH-12 already used.

## A. Which CH-14 checkpoint is the candidate

1. The candidate is **`outputs/final_rtdetr/train/weights/last.pt` after the 6th epoch**, the end of the frozen
   schedule (`final.json`: 6 epochs, 3 warmup epochs, linear decay). It is the same rule CH-11 used for
   `rtdetr_baseline` (its headline is `last.pt`) and the one CH-12 used to compare the families.
2. The per-epoch val_fast curve (`epoch0.pt` … `epoch5.pt`, `scripts/predict_ultra.py curve`) is reported, but it
   **does not choose the checkpoint**. `plan.md` §4 reserves val_fast for per-epoch curves, and picking the best epoch
   on val would tune on the split that the comparison below uses.
3. If the run does not finish all 6 epochs before the plan.md §7 cut-off (no new training after Tue 08:00), the
   candidate is the last fully written epoch checkpoint. This is recorded as a deviation and the fallback is kept,
   unless rule B passes anyway.

## B. CH-14 candidate vs the existing fallback

Fallback F: CH-11 `rtdetr_baseline` `last.pt` (sha256 `da5d3625…e872`) with the CH-13 pipeline
`configs/runs/final_inference.json`. val_full: mAP 0.2696, AP50 0.5069, AP75 0.2453, AP_S 0.1467, AP_M 0.2375,
AP_L 0.4502 (`configs/runs/final_inference.json` `evidence.val_full.selected`). Its test CSV
(sha256 `73f73e21…793b`) scored 0.2328849249 on the public LB (reported by the student).

Both models are evaluated on **val_full** (2,846 frames) with the project evaluator (`movd/evaluate.py`, maxDets
500) in the pipeline that would be submitted: `configs/runs/final_inference.json`, unchanged (no new tuning
on val_full or test). The single-pass numbers (Ultralytics predictor, conf 0.001, as in CH-11) are also reported as
the data-strategy gain over `rtdetr_baseline` (0.2577), but they do not decide.

The CH-14 candidate C **replaces F as the primary submission only if all of these hold**:

1. val_full mAP(C) ≥ mAP(F) + **0.0063**, the measured run-to-run spread (CH-10 replicate; CH-12 rule 3).
2. C leads F on **at least 5 of the 9 secondary checks**: AP50, AP75, AP_S, AP_M, AP_L and the 4 density bins of
   `results/ch-12-comparison/density.csv` (same bin edges). Every check C loses is reported.
3. AP_S(C) ≥ AP_S(F) − 0.003 (the CH-13 small-object guard).
4. The candidate passes the identity checks: trained on `train_final` (6,072 frames, ids sha256 `68bc424a…d42f`),
   6 completed epochs, non-smoke, and the checkpoint hash is recorded.

Otherwise F stays the primary submission. A mAP lead inside the spread is called a tie, not a win.

## C. Kaggle selections (G2c: 2 final selections, by internal validation)

* Primary selection: the winner of rule B.
* Second selection: the other model, if its validated test CSV exists. The public leaderboard score is recorded,
  but it does not override rule B (Kaggle Rules: the selection is by internal validation).
* No CSV is uploaded by Claude. Uploads and the final Kaggle selection are made by a student in the web UI.
* `val_roundtrip_ch13_combo_flip_tile_nms05.csv` (an accidental upload of a val round-trip file) is not a test
  submission and is never a candidate.
