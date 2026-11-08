---
title: Homework 2 -- Multiple Object Video Detection
author: Marko-Zenon and Badabuba
course: Artificial Intelligence · UCU
date: 2026-11-08
---

> Report scaffold and verified evidence snapshot at `c3873d2` (2026-11-08). TODO items require student authorship or named chunk evidence. AP values are fractions, not percentages. No discussion or conclusions are supplied here.

## 1. Introduction and objectives

**Student TODO:** Write the problem statement, aerial-video detection objectives, four-family comparison questions, and scope. Explain the intended accuracy–speed trade-off in your own words.

## 2. Dataset and exploratory analysis (CH-02, CH-06)

**Student TODO:** Describe the supplied clip-level split, eight merged classes, class imbalance, object sizes, density, sequence differences, occlusion and temporal redundancy. Justify subsampling using the tables below; distinguish target objects from repeated crowd annotations.

| Split | Clips | Frames | Target boxes | Unique ignore regions |
|---|---:|---:|---:|---:|
| Train | 56 | 24,201 | 1,105,516 | 63,843 |
| Validation | 7 | 2,846 | 114,132 | 3,995 |
| Test | 48 | 3,811 | Unknown (unlabelled) | Unknown |

Source: [CH-06 key numbers](results/ch-06-eda/key_numbers.json), `split_overview`; [class/size/density/sequence tables](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/tree/c3873d2/results/ch-06-eda/tables).

## 3. Preprocessing and ignored regions (CH-02, CH-04, CH-09)

**Student TODO:** Explain resizing, normalization and horizontal flips. Explain the frozen training mask with target pixels preserved and the unchanged crowd GT used in evaluation. Explain that constant filling is the chosen training heuristic; evaluate its limitations rather than claiming it removes every background-loss contribution.

![Ignore-mask pipeline check: original, ignore regions and training input](images/ch-15-ignore-masking-check.png)

Pipeline test figure, copied byte-for-byte from [CH-02 source](results/ch-02-protocol/figures/ignore_masking_check.png). **Student TODO:** inspect the legend and explain preservation of labelled target pixels.

## 4. Shared experimental protocol (CH-02, CH-03)

**Student TODO:** Explain train_rubric (2,800 frames), six epochs (16,800 images seen), seed 42, aspect-preserving 1344-pixel long side and recorded batch sizes. Distinguish val_fast (714 frames, curves/selection) from val_full (2,846 frames, headlines). Describe COCO IoUs 0.50:0.05:0.95, size bins and global top-500 cap. Cite configs/protocol.json and movd/evaluate.py. Separate legacy v0, smoke tests and rubric runs; do not compare their metrics as controlled experiments.

## 5. Two-stage detector: Faster R-CNN (CH-07)

**Student TODO:** Explain the pretrained ResNet-50-FPN v2 baseline and isolated RPN pre-NMS / RoI-positive-fraction changes. Interpret all six AP metrics, per-class AP, speed and three failure examples. Explain final-epoch selection on val_fast and single-seed limitations. Use curve.json for each epoch; no standalone corrected learning-curve figure is tracked yet.

### Verified six-epoch results

| Run | mAP | AP50 | AP75 | AP_S | AP_M | AP_L | FPS | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [frcnn_baseline](results/frcnn_baseline/metrics.json) | 0.1873 | 0.3808 | 0.1604 | 0.0913 | 0.1779 | 0.3452 | 18.69 | 53.50 |
| [frcnn_rpn_pre3000](results/frcnn_rpn_pre3000/metrics.json) | 0.1893 | 0.3847 | 0.1614 | 0.1083 | 0.1868 | 0.3107 | 14.85 | 67.35 |
| [frcnn_roi_pos50](results/frcnn_roi_pos50/metrics.json) | 0.1814 | 0.3643 | 0.1553 | 0.1004 | 0.1763 | 0.3109 | 17.48 | 57.19 |

Sources: each linked `metrics.json` → `val_full.metrics`; sibling `speed.json` → `inference`. All three: RTX 5070 Laptop GPU, Windows 11, Python 3.12.14, torch 2.14.1+cu130, torchvision 0.29.1+cu130, COCO_V1 initialization. Speed uses 10 warmups and 30 batch-one GPU-resident frames, including detector transforms and NMS; decoding and host transfer excluded. This is detector throughput, not end-to-end video FPS. [Protocol, runtimes and weight checksum](results/ch-07-frcnn/protocol.md).

### Per-class AP on val_full

| Class | Baseline | RPN pre3000 | RoI pos50 |
|---|---:|---:|---:|
| person | 0.2440 | 0.2450 | 0.2510 |
| bicycle | 0.1608 | 0.1950 | 0.1419 |
| car | 0.3774 | 0.3756 | 0.3902 |
| van | 0.2128 | 0.1725 | 0.2075 |
| truck | 0.1442 | 0.1589 | 0.1245 |
| tricycle | 0.1800 | 0.1981 | 0.1929 |
| bus | 0.0471 | 0.0569 | 0.0340 |
| motor | 0.1325 | 0.1125 | 0.1087 |

Source: same run metric files → `val_full.per_class[].AP`. Density-bin AP is stored under `val_full.density`; six epoch points under each run’s `curve.json` → `val_fast`.

## 6. One-stage anchor-based detector: RetinaNet (CH-08)

**Student TODO:** After CH-08 merges, explain the baseline, anchor geometry hypothesis, focal-loss alpha/gamma experiment and controlled comparison with Faster R-CNN. Add six val_full metrics, per-class AP, speed/latency, curves, failed runs and loss rationale. Currently only smoke artifacts are tracked; legacy v0 is not corrected rubric evidence.

## 7. One-stage anchor-free detector: YOLO11 (CH-09, CH-10)

**Student TODO:** After CH-09/CH-10 merge, describe the export/ignore and coordinate adapters, pretrained weights, baseline, assignment/loss/resolution variants and high-resolution-head compatibility decision. Add size-bucket and density results. YOLO11 evidence is pending on develop; do not substitute legacy YOLO versions.

## 8. Query-based detector: RT-DETR (CH-11)

**Student TODO:** Record the G9 memory/feasibility decision, exact weights and baseline plus tuning under comparable images-seen budget. Add project-evaluator curves against the CNN comparator and student analysis of convergence, small objects and dense scenes. No RT-DETR runs are tracked in this snapshot.

## 9. Cross-family comparison and final configuration (CH-12)

**Student TODO:** Replace pending cells only after all family evidence merges. Link results/summary.csv and configs/runs/final.json when available. Explain the final-family decision using accuracy, small-object AP, wall time and speed on comparable hardware. No final-family decision is available yet.

| Family | Rubric evidence on develop | Missing evidence owner |
|---|---|---|
| Faster R-CNN | Baseline + two isolated variants, six AP metrics, per-class/density AP, speed, curves and failures | Student interpretation / final comparison |
| RetinaNet | Smoke tests only | CH-08: full baseline, anchors and focal-loss tuning |
| YOLO11 | No tracked full runs | CH-09/CH-10 |
| RT-DETR | No tracked runs | CH-11, G9 |

## 10. Final pipeline and inference improvements (CH-13, CH-14)

**Student TODO:** Add one-change-at-a-time val_full comparisons for resolution, tiling, threshold/NMS/max_det, TTA and any validated ensemble. Include neutral/negative outcomes and speed costs. Link final_inference.json and final-training data-strategy comparison once produced. These artifacts are pending.

## 11. Kaggle submission and selection (CH-05, CH-14)

**Student TODO:** Explain original-pixel class score x y width height groups, classes 0–7, one row per test image and at most 500 detections. Cite the validator, lossless round-trip and upload log. Add final selection rationale based on internal validation; the first upload is not a final model or a baseline-beating result.

| Submission | val_full mAP | Kaggle public score | Status |
|---|---:|---:|---|
| Faster R-CNN RPN pre3000, epoch 6 | 0.1893354654 | 0.2011538266 | Accepted; below public baseline 0.22163 |

Source: [append-only submissions log](results/submissions.csv), [validation receipt](results/ch-05-submission/frcnn_rpn_pre3000_e6_test/validation.txt), [lossless real round-trip](results/ch-05-submission/roundtrip_frcnn_rpn_pre3000_e6_lossless.json). Test: 3,811 rows and 261,940 detections. Remaining: CH-14 baseline-beating score and two final selections; G1 exact UTC close time.

## 12. Limitations and reproducibility (CH-16)

**Student TODO:** Discuss one seed, CUDA nondeterminism, subset training, scene/domain shifts, speed-benchmark boundaries and unknown hidden Kaggle scorer equivalence. Add exact commands, package/weight versions and checksums from stored artifacts; perform the clean-environment check. Resolve G1 exact UTC deadline and G10 student authorship before freeze.

## 13. Discussion — student authored

**Student TODO:** Write your own evidence-based interpretation, trade-offs, unsuccessful hypotheses and answers to the objectives. Separate measured facts from explanations that were not experimentally tested.

## 14. Conclusions — student authored

**Student TODO:** Write your own conclusions only after final evidence and selection exist. State what the experiments support, remaining limitations and any justified incomplete rubric items.

## 15. References

**Student TODO:** Add original papers for Faster R-CNN, FPN, RetinaNet/focal loss, the specific YOLO11 implementation and RT-DETR; VisDrone and COCO evaluation sources; torchvision, pycocotools and Ultralytics documentation; exact pretrained-weight identifiers and package versions. Cite the Ultralytics AGPL-3.0 license. Verify bibliographic details and connect in-text citations before submission.

## Evidence and figure inventory

Figure convention: `images/<chunk>-<name>.png` (or the existing source extension), copied from `results/<chunk-or-run>/figures/`. Keep the source link and an explanatory student caption. CH-15 includes one pipeline test image only; the final figure selection/copying belongs to CH-16. No new charts were generated.

| Evidence | Verified source | Report image / action |
|---|---|---|
| Class distribution | [ch-06-eda/figures/class_distribution.png](results/ch-06-eda/figures/class_distribution.png) | TODO CH-16: select, copy and add student caption |
| Object sizes | [ch-06-eda/figures/object_sizes.png](results/ch-06-eda/figures/object_sizes.png) | TODO CH-16: select, copy and add student caption |
| Density | [ch-06-eda/figures/objects_per_frame.png](results/ch-06-eda/figures/objects_per_frame.png) | TODO CH-16: select, copy and add student caption |
| Sequence differences | [ch-06-eda/figures/per_sequence.png](results/ch-06-eda/figures/per_sequence.png) | TODO CH-16: select, copy and add student caption |
| Ground truth | [ch-06-eda/figures/annotated_frames.jpg](results/ch-06-eda/figures/annotated_frames.jpg) | TODO CH-16: select, copy and add student caption |
| Ignore masking check | [ch-02-protocol/figures/ignore_masking_check.png](results/ch-02-protocol/figures/ignore_masking_check.png) | `images/ch-15-ignore-masking-check.png` |
| Tiny-target failure | [ch-07-frcnn/figures/tiny_target_miss.png](results/ch-07-frcnn/figures/tiny_target_miss.png) | TODO CH-16: select, copy and add student caption |
| Car/van failure | [ch-07-frcnn/figures/car_van_confusion.png](results/ch-07-frcnn/figures/car_van_confusion.png) | TODO CH-16: select, copy and add student caption |
| Dense-frame failure | [ch-07-frcnn/figures/dense_frame_miss.png](results/ch-07-frcnn/figures/dense_frame_miss.png) | TODO CH-16: select, copy and add student caption |

**Student TODO:** use [failure metadata](results/ch-07-frcnn/failure_cases.csv) and [failure analysis](results/ch-07-frcnn/failure_analysis.md) to write your own interpretation. Failure display threshold 0.10 and same-class IoU check 0.50 are qualitative settings, separate from AP evaluation. Add a learning-curve figure from stored `curve.json` in the owning chunk; no standalone curve/comparison PNG is tracked here.

### Artifact completeness and provenance

| Evidence group | Stored material | Status / limit |
|---|---|---|
| CH-02 protocol | configs/protocol.json; split JSONs; data_facts.json; ignore test | Frozen protocol; Kaggle daily-limit field is stale, use current board G2b (5/day recorded) |
| CH-03 evaluator | sanity.json; speed_selftest.json; pytest.txt; CLI example | Synthetic/tool checks; never model leaderboard evidence |
| CH-06 EDA | key_numbers.json; tables/; figures/; run_stage1.py | Corrected EDA evidence; test labels unavailable |
| Three full Faster R-CNN runs | config.json, metrics.json, curve.json, speed.json, hardware.json, runtime.json per run; configs/runs/frcnn_*.json | Complete, six epochs each; one seed |
| Smoke groups | smoke_frcnn*, smoke_retina*, frcnn_*_smoke | Integration/preflight only, excluded from headline table |
| CH-05 submission | validator/prediction receipts; synthetic and real round-trips; submissions.csv | One accepted upload; final selection pending |
| CH-08 to CH-14 | See named section TODOs | Unmerged/pending work is not treated as completed evidence |

**Student TODO:** replace this snapshot with the final CH-16 traceability audit, retaining source links, failed experiments and exact package/weight versions. Check README commands against the frozen protocol before running them.
