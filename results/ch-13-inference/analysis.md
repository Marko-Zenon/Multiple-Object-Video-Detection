# CH-13 — final-pipeline inference experiments (evidence note)

Facts and numbers only; every number comes from a file in this folder. The interpretation and conclusions
for the report are left to the students (TODO prompts at the end).

## Setup

* Fixed reference R: CH-11 `rtdetr_baseline` `last.pt`, sha256 `da5d3625…e872` (= the CH-12 record), Ultralytics
  8.4.207 RT-DETR predictor, imgsz 1344, FP32, batch 1, decoder top-300 (query, class) pairs, conf > 0.001, no NMS,
  single pass. No training; the weights were never changed.
* Search space and selection rule: `selection_rule.md`, committed (2b658d0) before any CH-13 result. The
  combination variants and the merge-IoU clarification were committed (`variants.json` "combination_note")
  before the combinations were evaluated.
* Method: one GPU pass per view (`scripts/infer_val.py cache`) stores the raw decoder output (300 boxes x 8
  class scores per view); every decoding/merging variant is evaluated from that cache on the CPU with the
  project evaluator (`movd/evaluate.py`, maxDets 500).
* Identity checks: R from the cache reproduces CH-11 exactly: val_fast mAP 0.261704 (curve.json epoch 6) and
  val_full mAP 0.25765263737396193 (`results/rtdetr_baseline/metrics.json`). The decoder output equals
  Ultralytics' own output bit for bit (`tests/test_infer.py`, GPU test). The raw caches are deterministic
  (`determinism.json`: val_fast frames of two separate runs bitwise identical for all 6 views), and the live
  pipeline equals the cached path (`live_vs_cache.json`: 20/20 frames identical).

## val_fast screening (714 frames; R = 0.2617)

| Axis | Variant | mAP | Δ vs R | AP50 | AP_S | dets/frame | FPS | Rule |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 size | 1152 / 1536 / 1728 | 0.2431 / 0.2496 / 0.2351 | −0.0186 / −0.0121 / −0.0266 | 0.4599 / 0.4695 / 0.4517 | 0.1125 / 0.1400 / 0.1151 | 300 | 18.9 / 11.1 / 8.6 | fail |
| 2 tiling | 2x2 tiles + full, NMS 0.5 / 0.6 / 0.7 | 0.2681 / 0.2681 / 0.2672 | +0.0064 / +0.0064 / +0.0055 | 0.5027 / 0.5011 / 0.4956 | 0.1358 / 0.1366 / 0.1386 | 500 | 2.75 (0.5) | pass, winner NMS 0.5 |
| 3a conf | 0.01 / 0.05 | 0.2617 / 0.2607 | 0.0000 / −0.0010 | | | 300 / 160 | | fail |
| 3b top-k | 100 / 500 | 0.2425 / 0.2636 | −0.0192 / +0.0019 | | | 100 / 500 | 13.8 (500) | fail (below margin) |
| 3c post-hoc NMS | 0.7 / 0.5 | 0.2615 / 0.2607 | −0.0002 / −0.0010 | | | 249 / 202 | 13.7 (0.7) | fail |
| 4a flip TTA | NMS 0.6 / WBF 0.6 | 0.2686 / 0.2582 | +0.0069 / −0.0035 | 0.4960 / 0.4852 | 0.1337 / 0.1294 | 323 / 320 | 6.7 / 6.4 | NMS passes, winner |
| 4b two scales 1344+1536 | NMS 0.6 / WBF 0.6 | 0.2621 / 0.2504 | +0.0004 / −0.0113 | | 0.1449 / 0.1353 | 321 / 319 | 5.9 | fail |
| 5 ensemble + rtdetr_q500 | WBF 0.6 / NMS 0.6 | 0.2538 / 0.2609 | −0.0079 / −0.0008 | | | 437 / 441 | 6.3 / 6.9 | fail |
| rule 5 stack | flip + tiles, NMS 0.5 / 0.6 | **0.2725** / 0.2725 | +0.0108 / +0.0107 | 0.5083 / 0.5067 | 0.1439 / 0.1446 | 500 | 2.26 | kept (+0.0039 over flip) |

Full table with AP75, AP_M, AP_L, maxDets-100 mAP, density bins, detection counts and CPU time per frame:
`comparison.csv`; per class `per_class.csv`; per density bin `density.csv`; the rule applied: `selection.json`.
Figures: `figures/val_fast_deltas.png`, `figures/accuracy_vs_speed.png`, `figures/val_full_class_density.png`.

## val_full (2,846 frames; selected S = `combo_flip_tile_nms05`)

| Variant | mAP | AP50 | AP75 | AP_S | AP_M | AP_L | dets/frame | frames at 500 cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R (CH-12 baseline) | 0.2577 | 0.4775 | 0.2392 | 0.1296 | 0.2246 | 0.4538 | 300 | 0 |
| flip TTA only | 0.2660 | 0.4941 | 0.2472 | 0.1345 | 0.2302 | 0.4622 | 323 | 0 |
| tiling only | 0.2644 | 0.4991 | 0.2384 | 0.1425 | 0.2367 | 0.4404 | 500 | 2,771 |
| **S = flip + tiling** | **0.2696** | **0.5069** | 0.2453 | **0.1467** | 0.2375 | 0.4502 | 500 | 2,840 |

* S vs R on val_full: mAP +0.0120, AP50 +0.0293, AP75 +0.0061, AP_S +0.0171, AP_M +0.0129, AP_L −0.0036.
  The rule-7 veto did not trigger.
* Per class (AP, R → S): person 0.2748 → 0.2737, bicycle 0.2667 → 0.2584, car 0.4219 → 0.4350,
  van 0.3080 → 0.3210, truck 0.2296 → 0.2550, tricycle 0.1927 → 0.2469, bus 0.2347 → 0.2366, motor 0.1328 → 0.1305.
* Per density bin (mAP, R → S): 0-19 targets 0.1284 → 0.1994; 20-49 0.2611 → 0.2816; 50-99 0.2749 → 0.2859;
  ≥ 100 0.3679 → 0.3614.
* With tiling the 500-per-image cap binds on 2,840 of 2,846 frames (R never exceeds its 300-query cap).

## Speed and memory (RTX 5070 Laptop GPU, same session, `speed/`)

| Pipeline | passes / frame | FPS end to end | 3,811 test frames |
|---|---:|---:|---:|
| R | 1 | 13.82 (start) / 13.68 (end of session) | 4.6 min |
| flip TTA | 2 | 6.72 | 9.5 min |
| tiling | 5 | 2.75 | 23.1 min |
| **S** | 6 | **2.26** | **28.1 min** |

S is about 6.1x slower than R and within the predeclared 1.6 FPS limit. Peak CUDA memory allocated stays
0.65 GiB at imgsz 1344 for every view (0.81 GiB at 1536, 0.98 GiB at 1728; `passes/*.json`). GPU wall-clock
for all CH-13 caches: val_fast about 10 min, val_full about 23 min.

## Submission format

The selected val_full predictions went through the CH-05 lossless writer and validator
(`scripts/submission_roundtrip.py`): 1,422,988 detections, 0 rejected, ≤ 500 per row, validator problems
none. The CH-05 round-trip tolerance (1e-6) is exceeded (max 9.9e-6, `roundtrip_combo_flip_tile_nms05.json`)
because the writer clips the 64,496 boxes that extend outside the frame; clipping the direct predictions the
same way first gives a difference of exactly 0.0 (`roundtrip_clipping_check.json`). Clipped val_full mAP:
0.26963 (unclipped 0.26963). No test inference and no Kaggle upload were made in CH-13.

## Deviations

* Tuning on val_fast (user instruction) instead of val_full as plan.md CH-13 scope reads; val_full was used
  only for R, S and the two adopted single changes.
* The stacked pipeline has one merge NMS; the two axis winners used IoU 0.5 and 0.6, so both were evaluated
  (recorded before the results); 0.5 won by 0.0001.
* The strict "flip every view" stack (10 passes) was excluded by the speed rule from measured pass times
  (1.34 FPS), not by a measured end-to-end benchmark.
* The ensemble was tested with the two CH-11 rubric-budget checkpoints (`rtdetr_q500` as the second member).

## TODO for the students (report prose)

* TODO: why does flip TTA help with NMS merging but hurt with WBF here?
* TODO: why does a larger single input size (1536, 1728) hurt a model trained at 1344, while tiling (a 1.67x
  zoom on each tile) helps? Relate it to the stretched-square input and the 300-query cap.
* TODO: interpret the per-density change (large gain on sparse frames, small loss on the densest bin) and the
  500 cap binding on almost every frame with tiling.
* TODO: is a 6x slower inference acceptable for this task (offline Kaggle scoring vs a real-time drone use case)?
