# OPT-05 selection rule (committed before any OPT-05 candidate result)

OPT-05 is an optional post-plan experiment, the last optimization cycle, requested by student Badabuba on 2026-11-10.
Worker: Claude. Claude uploads nothing to Kaggle; uploads and the two final Kaggle selections stay with the students.
The grid, the gates and the stop conditions are never changed after a result is seen. No further experiment is run under any
outcome unless Badabuba explicitly approves one.

When this file was committed, no OPT-05 merge had been evaluated. The only numbers in it are values already registered on
`develop` (CH-14b Model A evidence, OPT-04 FB evidence).

## Hypothesis

FB (OPT-04 fallback: RT-DETR-l, imgsz 1600 for training and all 6 CH-13 pipeline passes, epoch 1) is worse than Model A overall
(val_fast 0.2761 vs 0.2892) but better on small objects (val_fast AP_S 0.1577 vs 0.1427). Its losses are on bus, van and large objects.
If only FB's **small** predicted boxes are added to Model A's full prediction set, Model A keeps its medium/large strength and
gains part of FB's small-object recall and localization, so overall mAP rises.

## Inputs (read in place, sha256-checked before every use)

| Input | Path (under `C:/UCU/AI/hw2/`) | sha256 |
|---|---|---|
| Model A checkpoint (CH-14 `final_rtdetr` epoch0.pt) | `MOVD-ch14/outputs/final_rtdetr/train/weights/epoch0.pt` | `6ff4a430188405b5d13356872c63f141c4d3dbc7b747983cdf733ce386aa5b0b` |
| FB checkpoint (OPT-04 `opt04_rtdetrl1600_e1` epoch0.pt) | `MOVD-opt04/outputs/opt04_rtdetrl1600_e1/train/weights/epoch0.pt` | `e06a1a4b3a4213147fe98083b5a8d3cd50aaa74dc7c38c72c0047e900acca33f` |
| Model A val_fast (CH-13 pipeline) | `MOVD-ch14b/outputs/ch14b/stageA_epoch1/val_fast/pipeline_predictions.json` | `87137d7df1478253733cf0f1d5d611067c5c22252a63df0dfee5b6da67275a0e` |
| Model A val_full | `MOVD-ch14b/outputs/ch14b/stageA_epoch1/val_full/pipeline_predictions.json` | `fccda2b3cc14f335468e872e95981592f4e02e5510d329a6a2c808916da7036c` |
| Model A test | `MOVD-ch14b/outputs/ch14b/stageA_epoch1/test/test_predictions.json` | `5f9c4913378dd5b7d740d70827255675a2035801c0dba851656a70806187930e` |
| FB val_fast (1600 pipeline, `configs/runs/opt04_inference_1600.json` sha256 `2b47bf2d…2041`) | `MOVD-opt04/outputs/opt04/FB_rtdetrl1600_e1/val_fast/pipeline_predictions.json` | `f0c86d7077d9d05d3f4fd50da508d844819479572d90b2c6120d6a30b43107e0` |
| FB val_full / test | not yet produced; only generated (GPU, same checkpoint and pipeline, `scripts/opt04_infer.py`) if the gate below allows | recorded when produced |

FB is not retrained. Model A's predictions are not regenerated. Frame splits and image sizes come from `movd.data`
(`val_fast` 714 frames, `val_full` 2,846, frozen in CH-02). Image sizes in val: 1344×756, 1904×1071, 2688×1512, 2720×1530, 3840×2160.

## Merge (deterministic, `movd/specialist.py`)

Per frame:

1. Start with **all** Model A detections, records unchanged (same box, class and score; no re-scoring).
2. Take FB detections whose predicted box is *small* under the candidate's size rule (below). All other FB detections are discarded.
   Each kept FB score is multiplied by the candidate's fixed multiplier `m`.
3. Union, sort stably by score (descending; ties: Model A before FB, then the record's position in its source list).
4. Class-wise NMS at the candidate's IoU (torchvision `batched_nms`, the call used by `movd.infer.nms` and OPT-02).
5. Keep the top 500 by score (global per-frame cap, as in the CH-13 pipeline and the evaluator).
6. Boxes are passed through in original-image pixel `xywh` exactly as stored (no coordinate round trip).

Model A's cached boxes already went through class-wise NMS at 0.5, so with NMS IoU ≥ 0.5 an A box is never removed by another A box.
An A box can leave the output only if an eligible FB box of the same class with a higher (scaled) score overlaps it above the IoU,
or if it falls below the top-500 cap. With no eligible FB box the output is exactly Model A's predictions (checked by `verify`).

## Size rules (predeclared, not tuned per class)

* **`coco32`** — COCO small: predicted box area `w·h < 32² = 1024` px² in original image pixels.
  This is the definition used by AP_small.
* **`in32`** — small at Model A's network input: Model A sees every frame as a stretched 1344×1344 square, so the box is small
  if `(w·1344/W)·(h·1344/H) < 32²`, i.e. the normalized area `w·h/(W·H) < (32/1344)² ≈ 5.67e-4`.
  For 1344×756 frames this is `w·h < 576` px² (stricter than coco32); for 3840×2160 it is `w·h < 4,702` px² (looser).

## Candidate grid (8, frozen; `candidates.json`)

| id | size rule | FB score multiplier m | NMS IoU |
|---|---|---|---|
| s1 | coco32 | 1.00 | 0.5 |
| s2 | coco32 | 1.00 | 0.6 |
| s3 | coco32 | 0.75 | 0.5 |
| s4 | coco32 | 0.75 | 0.6 |
| s5 | in32 | 1.00 | 0.5 |
| s6 | in32 | 1.00 | 0.6 |
| s7 | in32 | 0.75 | 0.5 |
| s8 | in32 | 0.75 | 0.6 |

Why these values: 1.00 trusts FB's scores as they are; 0.75 is the conservative weight that was best for the second model in OPT-02.
NMS 0.5 is the frozen CH-13 value; 0.6 was the better value in OPT-02. Both are ≥ 0.5, so Model A boxes never suppress each other.
Nothing else varies. No per-class rule. No full-FB candidate.

## Reference (Model A, exact stored values)

From `results/ch-14b-rtdetr-short-finetune/stageA_epoch1/{val_fast,val_full}/pipeline/metrics.json`:

| Split | mAP | AP50 | AP75 | AP_S | AP_M | AP_L | 0-19 | 20-49 | 50-99 | ≥100 |
|---|---|---|---|---|---|---|---|---|---|---|
| val_fast | 0.2892166441 | 0.5397315913 | 0.2662824512 | 0.1427339181 | 0.2593527694 | 0.4901336150 | 0.2132257513 | 0.3018068258 | 0.3102638579 | 0.3515017523 |
| val_full | 0.2859619320 | 0.5358305828 | 0.2600225327 | 0.1503772481 | 0.2568255865 | 0.4812468780 | 0.2081171391 | 0.2980836750 | 0.3072906016 | 0.3497290291 |

Model A width ≤ 1920 val_fast subset (322 frames, computed by `scripts/opt04_gates.py` in OPT-04 and recomputed here by the same
`movd.evaluate.evaluate(image_ids=…)` call): 0.3120912393562929.

All metrics come from the project evaluator (`movd/evaluate.py`, maxDets 500, frozen crowd/ignore policy), compared unrounded.

## Gate V (val_fast screen → val_full)

A candidate passes only if **all** hold:

| Check | Threshold |
|---|---|
| mAP | ≥ 0.2922166441 (A + 0.003) |
| AP_small | ≥ 0.1427339181 (no decrease) |
| ≥100-object density bin mAP | ≥ 0.3465017523 (A − 0.005) |
| width ≤ 1920 subset mAP | ≥ Model A's on the same frames (0.3120912394, no regression) |
| determinism | a second, independent merge + evaluation gives the same predictions sha256 and identical metrics |

**Selection:** among passing candidates, the highest val_fast mAP; on an exact tie, the lower id. Exactly one candidate or none.

## Gate F (val_full, run once, only for the selected candidate)

FB val_full predictions are produced once with the FB checkpoint and `configs/runs/opt04_inference_1600.json` (RTX 5070, claimed on the board first),
merged with Model A's stored val_full predictions by the frozen rule of the selected candidate, evaluated once.

| Check | Threshold |
|---|---|
| mAP | ≥ 0.2890 (A + 0.003, literal) |
| AP_small | ≥ 0.1504 (literal) |
| ≥100 bin mAP | ≥ 0.3447290291 (A − 0.005) |
| secondary checks improved or tied (X ≥ A, unrounded) | ≥ 5 of 9: AP50, AP75, AP_S, AP_M, AP_L, 4 density bins |

## Gate T (test CSV, only if Gate F passes)

FB test predictions from the same checkpoint and pipeline; merged with Model A's stored test predictions by the frozen rule;
lossless CSV via `movd.submission`; 3,811 rows; `scripts/validate_submission.py` and `scripts/check_submission_independent.py`
exit 0; 0 rows differ from the merged predictions; path, size, sha256, both checkpoint hashes, runtime and a suggested Kaggle description recorded.
**Not uploaded by Claude.**

## Stop conditions

* **V0** no candidate passes Gate V → stop: no val_full, no GPU, no test CSV; record a negative result; close OPT-05.
* **F0** the selected candidate fails Gate F → stop: no test inference, no CSV; record a negative result; close OPT-05.
* **T0** Gate T fails for a technical reason → fix the pipeline (never the rule) or record the failure; no CSV is called upload-ready.
* Any input hash mismatch → stop and report; nothing is evaluated on unverified inputs.

## Final two selections

| Outcome | Primary | Secondary |
|---|---|---|
| V0 or F0 (negative) | **Model A** `e587c68b…0da7` (val_full 0.2860, public 0.2774301362) | **F** `73f73e21…793b` (val_full 0.2696, public 0.2328849249) |
| Gate T passes | Model A (protected; the specialist is not promoted by val only) | the specialist CSV, replacing F as the hedge, after the students upload it |

The public score never promotes a candidate and never changes the rule.

## Known limitations

* **Cached predictions are already capped.** Both Model A's and FB's stored pipeline predictions already had class-wise NMS 0.5 and the
  global top-500 cap applied inside their own pipelines (no pre-cap predictions exist). 599/714 Model A val_fast frames are already at 500
  boxes (OPT-02), so every added FB box pushes out one of Model A's lowest-scored boxes in those frames, and FB boxes that FB's own cap
  removed cannot be recovered.
* val_fast (714) is a subset of val_full (2,846); the candidate is chosen on val_fast. One seed per model.
* The size rule uses the *predicted* box; COCO AP_S uses the GT area, so an added FB box may match a GT outside the small range.
* val contains a 3840-wide clip that test does not have; the width ≤ 1920 subset guard addresses this partly.
* FB is not protocol-comparable at 1344 (pre-approved resolution exception from OPT-04).
