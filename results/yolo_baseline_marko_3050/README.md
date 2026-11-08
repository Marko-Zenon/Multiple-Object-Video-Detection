# YOLO11s laptop baseline: verified CH-09 evidence

Owner: Marko-Zenon. Full-baseline resume claim `a20a01b` was pushed to develop before implementation/training. Training source commit `80781c4427ef3fd56a71a5fa0bb22956a9635418`, branch `ch-09-yolo`. Preserves CPU `8e48881`, smoke `d84bf00` and original run directories.

## Configuration and protocol

Run: `yolo_baseline_marko_3050`; config `configs/runs/yolo_baseline_marko_3050.json`. Official COCO-pretrained YOLO11s, 1344 pixels, all 2,800 train_rubric frames, six epochs, batch 1, AMP true, workers 0, cache false, seed 42 and frozen flip-only augmentation. Exactly 16,800 images and 16,800 post-backward batches completed without interruption. Gradient accumulation nbs64 is retained; batches are not optimizer updates.

Differences from the provisional shared baseline: unique run ID, batch 4 -> 1, workers 2 -> 0, cache explicitly false. Shared config, split, resolution, seed and six-epoch image budget were unchanged. Native optimizer=auto selected AdamW (lr 0.000833). Pinned Ultralytics 8.4.207 remapped 5/8 pretrained classifier rows by class name and logged 499/499 transferred items. No new model or hyperparameter experiment was introduced.

Dataset check passed: train 24,201/56 clips, val 2,846/7, test 3,811/48. Export recheck passed: train 2,800/137,756 targets, val_fast 714/28,685, all class IDs 0-7, zero duplicates and <=0.00288 px coordinate error. Ignore mask and JPEG pixel checks passed. See results/ch-09-yolo/marko_baseline_export/. Official pretrained SHA256: `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`. No package installation in this follow-up.

## Measured training losses and project learning curve

| Epoch | Images / batches | Mean box loss | Mean cls loss | Mean DFL loss | Project val_fast mAP |
|---|---:|---:|---:|---:|---:|
| 1 | 2,800 | 1.60578 | 1.18525 | 1.16181 | 0.17641914 |
| 2 | 2,800 | 1.48538 | 0.9385 | 1.11623 | 0.18454328 |
| 3 | 2,800 | 1.40437 | 0.83341 | 1.08214 | 0.15629284 |
| 4 | 2,800 | 1.33393 | 0.75662 | 1.05468 | 0.16096419 |
| 5 | 2,800 | 1.25737 | 0.68628 | 1.02389 | 0.14795409 |
| 6 | 2,800 | 1.17565 | 0.62598 | 0.99509 | 0.14405777 |

All 16,800 observed scalar batch losses were finite (range 1.53873467–12.23427010). Six saved checkpoints and six project-evaluator curve points are verified. The curve peaked at epoch 2; the headline uses the final checkpoint per the runbook, without checkpoint selection or extra training.

## Final val_full: project evaluator only

Final checkpoint: outputs/yolo_baseline_marko_3050/train/weights/last.pt. Evaluation: 2,846 original validation images; conf 0.001, NMS IoU 0.7, maxDets500, FP32. Crowd copies/ignore semantics retained by movd.evaluate. No Ultralytics internal metric is substituted.

| Metric | Value |
|---|---:|
| mAP | 0.1438917717 |
| AP50 | 0.2925317447 |
| AP75 | 0.1192264113 |
| AP_small | 0.0865789889 |
| AP_medium | 0.1434272936 |
| AP_large | 0.2554700319 |

635,095 predictions; no out-of-split detections or images over the shared 500 cap.

| Class | AP |
|---|---:|
| person | 0.25058446 |
| bicycle | 0.15652080 |
| car | 0.33808017 |
| van | 0.10684102 |
| truck | 0.05368041 |
| tricycle | 0.09803008 |
| bus | 0.03043726 |
| motor | 0.11695997 |

| Density bin (targets/frame) | Frames | mAP | AP50 |
|---|---:|---:|---:|
| 0-19 | 977 | 0.04633170 | 0.06337130 |
| 20-49 | 1062 | 0.14153213 | 0.26063455 |
| 50-99 | 618 | 0.18428975 | 0.40266966 |
| >=100 | 189 | 0.28374089 | 0.58141088 |

## Timing and resources

FP32 inference benchmark: **36.3745 FPS / 27.4918 ms/image** on RTX 3050 Laptop. 30 preloaded original BGR frames, 10 warmups, batch1, synchronized timing, includes letterbox/normalization/NMS/rescaling, excludes disk I/O. Full prediction time (including image reads) was 284.3 seconds; this is a different timing scope.

| Phase | Monitored wall seconds | Peak process RSS GiB | Minimum available RAM GiB | Peak sampled total GPU MiB |
|---|---:|---:|---:|---:|
| train | 2007.217 | 2.777 | 3.278 | 1849 |
| curve | 718.295 | 2.137 | 4.181 | 485 |
| final | 575.680 | 2.944 | 3.829 | 485 |
| failures | 8.170 | 0.646 | 5.533 | 0 |
| tests | 34.857 | 2.205 | 4.557 | 0 |

Training CLI time: 2002.990 s (33.38 min). Peak CUDA allocated/reserved: 1,454,161,408 / 1,625,292,800 bytes (1.354 / 1.514 GiB). RSS/device totals are one-second samples and may miss sub-second peaks. All monitored jobs exited 0, with no OOM or low-RAM guard trigger. Swap was already heavily used, increased during evaluation and was not reconfigured. Full samples and console logs stay under outputs/ch09_marko_baseline_20261108/.

## Retained checkpoint/resume state

| Path under outputs/yolo_baseline_marko_3050/train/weights/ | Bytes | State |
|---|---:|---|
| best.pt | 19,284,881 | Completed checkpoint, optimizer stripped by Ultralytics |
| epoch0.pt | 94,823,715 | Optimizer/scaler retained, CPU load verified |
| epoch1.pt | 94,823,779 | Optimizer/scaler retained, CPU load verified |
| epoch2.pt | 94,823,907 | Optimizer/scaler retained, CPU load verified |
| epoch3.pt | 94,823,971 | Optimizer/scaler retained, CPU load verified |
| epoch4.pt | 94,824,035 | Optimizer/scaler retained, CPU load verified |
| epoch5.pt | 94,959,395 | Optimizer/scaler retained, CPU load verified |
| last.pt | 19,284,881 | Completed checkpoint, optimizer stripped by Ultralytics |

All six epoch checkpoints contain the expected epoch, 255 optimizer entries and scaler scale512. The earlier smoke verified actual native epoch-boundary recovery; the full run required no resume. Completed last.pt cannot be resumed as unfinished training. If interrupted before completion, use the identical config and `train_ultra.py --device 0 --observe --resume`; unsaved current-epoch batches repeat and must be recorded. Never launch a new experiment into an existing run directory.

## Failure and comparison evidence

Three source-backed examples are stored in failure_cases.csv and figures/{tiny_target_miss,car_van_confusion,dense_frame_miss}.png. Thresholds match CH-07 (score>=0.10, same-class IoU>=0.50). The figures and curve were decoded and visually checked. Students supply their interpretations.

frcnn_comparison.csv contains all three CH-07 runs and this YOLO run from stored project val_full metrics. YOLO final mAP 0.14389177 is below FRCNN baseline 0.18732811 and selected FRCNN RPN 0.18933547. The comparable budget is 2,800 frames x six epochs, frozen splits/ignore policy and evaluator maxDets500. Hardware, detector training defaults and timing input placement differ. FRCNN speed used an RTX 5070 with GPU-resident tensors; YOLO speed uses an RTX 3050 with CPU BGR preprocessing/transfer. No architecture-only speedup claim or final-family selection is made.

## Validation, review and transfer

CH-09 tests: 39 passed before launch. Final full CPU suite: 149 passed, 2 skipped (GPU benchmark hidden; optional generated sample absent). Separate GPU benchmark integration: 1 passed. Checkpoint/evidence audit PASS, scoped CH-09 Ruff/format (10 files) and git diff --check PASS. Full-repository lint finds pre-existing UP035 in movd/data.py and PLW1510 in tests/test_data.py; full formatting flags six unchanged files (lint_summary.json). Those unrelated files were preserved. CUDA availability/NMS and official pretrained hash passed before launch. Protected website files, notebook, shared protocol/configs, CH-08/10/11 and other experiment artifacts were not edited.

No Kaggle upload, full-run restart, material protocol deviation or external transfer occurred. No completion PR was opened and CH-09 is not DONE. Technical baseline/evaluation acceptance evidence is available for review. Artifact destination/authorization and recipient verification remain pending. Notebook subsection integration from the runbook remains pending; no notebook lock was taken or notebook edited.

artifact_manifest.json records relative paths, bytes and SHA256; artifact_manifest.sha256 and transfer_files.txt support verified copying. Inventory includes full-run checkpoints/predictions/logs/evidence plus preserved laptop smoke artifacts. Large outputs remain ignored. After Badabuba confirms an SSH/private-cloud destination, copy into a fresh directory (for example rsync --files-from=transfer_files.txt over SSH), copy the manifest separately, and have Badabuba run sha256sum -c artifact_manifest.sha256. Do not report transfer complete until he verifies every file. Historical smoke manifests retain their original source hashes; the new inventory is the current transfer-integrity manifest.
