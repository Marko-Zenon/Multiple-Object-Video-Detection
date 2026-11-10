# CH-14 — final training, final comparison and submission selection

All numbers below are read from the files in this folder, `results/final_rtdetr/` and `results/submissions.csv`. All APs come from
the project evaluator (`movd/evaluate.py`: val_full, 2,846 frames, maxDets 500). Rebuild the tables, decision and figures with
`python results/ch-14-final/make_evidence.py --hash-checkpoints` (CPU only). The conclusions are written by the students; see the TODOs at the end.

## 1. What was trained

| item | value | source |
|---|---|---|
| run config | `configs/runs/final_rtdetr.json` = `configs/runs/final.json` `run_config` + `"train_split": "train_final"` | `tests/test_ch14_final.py` |
| initialization | official COCO-pretrained Ultralytics `rtdetr-l.pt`, sha256 `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f` (= `final.json` `pretrained_sha256`) | `results/final_rtdetr/config.json` |
| training list | `train_final`: every 4th frame of each of the 56 train clips, 6,072 unique frames; id-list sha256 `68bc424a08eeb68460bab09cbdbf90b53f8e4e9b80eb3b1d1e3c8770d2add42f` (= `final.json`); split file `configs/splits/train_final.json` sha256 `63732dd60f6bbff00a32d436e4f56557b9b01b14f59fd4a05658103ad0ffcafa`; 0 val/test clips | `scripts/make_splits.py --check`, tests |
| export | masked with the CH-02 policy: 4,538 of 6,072 frames masked (16,036 ignore regions), 277,673 labels, max box error 0.002 px, max fill MAE 1.64 | `export/export_check.json`, `export/figures/export_check.png` |
| budget | 6 epochs x 6,072 = 36,432 images seen (loader: 6,072 samples, 36,432 batches), imgsz 1344, batch 1, AMP, seed 42, flip only, optimizer auto | `results/final_rtdetr/runtime.json`, `config.json` |
| runtime | 8,988.9 s = 2.50 h in one process (no resume needed), epochs 1,461-1,560 s, peak CUDA 3.55e9 B allocated; RTX 5070 Laptop 8 GB | `epochs.csv`, `runtime.json`, `hardware.json` |
| checkpoints | `outputs/final_rtdetr/train/weights/` (git-ignored); candidate `last.pt` sha256 `a85d2741f1891df7ee536a99e98f1e02d2c188591a1b9634728e5930469d482b` | `checkpoints.json` |

The manifest fix: `scripts/train_ultra.py` used to read `train_rubric_manifest.json` whatever `data.yaml` trained on. It now calls
`movd.ultra.train_manifest`, which takes the run config's `train_split` (default `train_rubric`, so old configs behave as before)
and refuses to start unless `data.yaml`, `<split>_manifest.json`, the list file and the frozen split file all name the same frames.

The smoke test ran on the real `train_final` path (`results/final_rtdetr_smoke/`, `smoke_check/`): 50 frames x 2 epochs, finite losses,
3.55e9 B peak, stop after epoch 1, then resume with the optimizer state, then inference and evaluation in both modes.

## 2. Results (val_full)

| model | mode | mAP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---|---|---:|---:|---:|---:|---:|---:|
| CH-14 final_rtdetr `last.pt` | single pass | 0.2359 | 0.4479 | 0.2130 | 0.0883 | 0.2103 | 0.4003 |
| CH-14 final_rtdetr `last.pt` | CH-13 pipeline | 0.2372 | 0.4648 | 0.2086 | 0.1490 | 0.2231 | 0.3939 |
| fallback rtdetr_baseline (CH-11) | single pass | 0.2577 | 0.4775 | 0.2392 | 0.1296 | 0.2246 | 0.4538 |
| fallback rtdetr_baseline (CH-11) | CH-13 pipeline | **0.2696** | 0.5069 | 0.2453 | 0.1467 | 0.2375 | 0.4502 |

Per class: `per_class.csv`, `figures/per_class.png`. Density bins: `density.csv`, `figures/density.png`.
The single-pass CH-14 number from `scripts/ch14_infer.py` matches `scripts/predict_ultra.py final` exactly (0.2359,
`results/final_rtdetr/metrics.json`). That is the identity check that the single-pass mode is the unchanged CH-11 method.

Per-epoch val_fast curve (single pass; reported, it does not choose the checkpoint, `selection_rule.md` A.2):

| epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| final_rtdetr | 0.2790 | 0.2585 | 0.2614 | 0.2609 | 0.2531 | 0.2381 |
| rtdetr_baseline | 0.2224 | 0.2594 | 0.2691 | 0.2527 | 0.2553 | 0.2617 |

Same-session speed (`speed/session.json`, first 30 val_full frames preloaded, FP32, batch 1, RTX 5070): both checkpoints run at
13.83 FPS single pass and 2.29 / 2.27 FPS with the CH-13 pipeline (about 28 min for the 3,811 test frames), peak 0.70 / 0.80 GB
allocated. The architecture is the same, so the speed is the same.

## 3. Decision (rule B of `selection_rule.md`, committed before any CH-14 result)

`selection.json`: lead of C over F on val_full mAP (CH-13 pipeline) = **-0.0324** (needs >= +0.0063); secondary checks won **1/9**
(only AP_small, 0.1490 vs 0.1467; needs >= 5); AP_small guard passed; identity passed. **The fallback stays the primary submission**:
CH-11 `rtdetr_baseline` + CH-13 pipeline, `rtdetr_baseline_ch13_fliptile_nms05_test.csv`, sha256
`73f73e219dfbf3b674277703f7314d14c316d0800bcd046634aaecd27f2b793b`, public LB 0.2328849249 (reported by the student).

Because CH-14 lost, no CH-14 test inference was run. The already validated fallback CSV is reused unchanged (`final_submission.json`).
This is a negative result and it is kept: the larger training list did not improve the final checkpoint at this schedule.

Facts relevant to the explanation (no conclusion drawn here):
* final_rtdetr's val_fast curve is highest after epoch 1 (0.2790) and falls over epochs 2-6 to 0.2381. rtdetr_baseline's curve
  rises to 0.2691 at epoch 3 and ends at 0.2617.
* The schedule is the frozen one (Ultralytics counts warmup and the linear decay in epochs), so with 2.17x more iterations per epoch,
  final_rtdetr made 2.17x more optimizer steps at every point of the same lr curve.
* The single-pass training losses kept falling in every epoch (`epochs.csv`: GIoU 0.749 -> 0.390).
* Not evaluated on val_full, by the rule: epoch checkpoints other than `last.pt`. Choosing epoch 1 after seeing the curve would be
  selection on val, so it is not done here.

## 4. Submission checks

The fallback CSV passed both validators: `scripts/validate_submission.py` (project, `fallback/project_validator.txt`) and
`scripts/check_submission_independent.py` (standard library only, no `movd`, `fallback/independent_validator.json`).
Results: 3,811 rows in `test_images.csv` order, header `image_id,PredictionString`, 1,902,580 groups, at most 500 per row, classes
0-7, finite scores and boxes inside the frame, lossless `repr` numbers, 0 rows differing from the raw test predictions, and
checkpoint sha256 `da5d3625…e872` recorded in the test run. The accidental upload `val_roundtrip_ch13_combo_flip_tile_nms05.csv` is
logged in `results/submissions.csv` as NOT A CANDIDATE. It fails the project validator (2,846 val rows, all 3,811 test ids
missing; `fallback/accidental_upload_project_validator.txt`), so CH-14 acceptance criterion 3 ("every uploaded file passed the
validator") is not literally met. A student must record how this upload is treated before CH-14 can be DONE.

## 5. Student TODOs (conclusions are written by the students)

- TODO (students): explain why 2.17x more distinct frames, under the same epoch-based schedule, ended lower on val_full than the
  rubric-budget model. Use the curve, the per-class table (truck improved, bus and van dropped) and the density bins.
- TODO (students): discuss whether the result is a data effect or a schedule/step-count effect, and what experiment would separate them.
- TODO (students): choose the two final Kaggle selections by internal validation (G2c) and record the rationale. Only two validated
  test uploads exist: the fallback (val_full 0.2696, public 0.2328849249) and CH-07 Faster R-CNN (val_full 0.1893, public 0.2011538266).
