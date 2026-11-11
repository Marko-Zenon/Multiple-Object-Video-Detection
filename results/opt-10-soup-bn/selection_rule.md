# OPT-10 selection rule — multi-run weight soup + BatchNorm re-estimation on E6 (frozen before any result)

Written and committed **before** any OPT-10 diagnostic or candidate was run on any split, before any BatchNorm
statistic was re-estimated and before any replicate was trained. Not changed after a result is seen; later notes are
appended with a timestamp and never change an earlier rule. Requested and approved by Badabuba (Nazar) on 2026-11-11
(plan file `pasted-content-id-f4af-act-as-harmonic-ocean.md`, with the review corrections of the same day).
Worker: Claude. Code: `movd/opt10.py` (grid, gates, frame lists, weight averaging, BatchNorm tools),
`scripts/opt10_soup.py` (checkpoints, raw caches, diagnostics), `scripts/opt10_loop.py` (stages, gate).

## Protected baseline: E6 (OPT-08 `e6_combination` k1), never overwritten

| Item | Value |
|---|---|
| Pipeline | soup w080 -> CH-13 merge (`configs/runs/final_inference.json`: full + flip + 4 tiles at 1344, class-wise NMS 0.5, top 500) with inner tile-border boxes dropped (factor 0, margin 0.005) -> five-frame persistence boost (IoU 0.5, strength 1). `movd.opt09.E6`, reused unchanged for every OPT-10 candidate |
| Soup w080 | 0.8 x Model A `epoch0.pt` (`6ff4a430…5b0b`) + 0.2 x `epoch1.pt` (`5594fb0e…5799`); `MOVD-opt08/outputs/opt08/soup/soup_w080.pt` sha256 `2d60d3fa66d9fd37032bcaaaeb297e98510217b600b5809f73873b36360cd939` |
| val_full predictions | `MOVD-opt08/outputs/opt08/e6_combination/k1/val_full.json` sha256 `bdb2806edad09fa093d5b59369780a8dbbb6fea4758522c72c86b8c8e98cad52`; val_full mAP 0.29658283872855484, discovery 0.3344880450293952, confirmation 0.27089469511565134, width <= 1920 0.31812638768024115 |
| CSVs | E6 `opt08_e6_combination_k1_test.csv` sha256 `35a225c905ac56ed86532314046821d10085f73781666ee843fb4c6d28127c35` (public 0.2947349001, user-reported); E4 `opt08_e4_checkpoint_soup_s1_test.csv` sha256 `7dc9b318e354d4839b5dffae8f5973dd7da234fdbeb3c5f84057694878852014` (public 0.2913788894). Both are the Kaggle final selections and are only read |
| Raw caches | `MOVD-opt08/outputs/opt08/soup/raw/w080/{discovery,confirmation,test}/*.npz`, read only |

`scripts/opt10_loop.py verify` checks every hash above (the OPT-09 protected list), that the frozen files equal
`origin/develop`, that the OPT-10 code path rebuilds the E6 val_full predictions byte-identically, and that
`movd.opt10.weighted_state` with weights (0.8, 0.2) on the two E6 checkpoints gives the tensors of soup w080 exactly.
It runs before the first result and again at the end of every session. Any mismatch: stop.

## Hypothesis

Model A's checkpoint is one noisy draw of its training recipe: Ultralytics' weight EMA is inactive after 915 updates
(decay 0.37), and the 122 BatchNorm layers were trained at batch size 1 with momentum 0.03, so their inference
statistics average only the last ~33 training frames. E6's two-checkpoint soup averaged the weights and those
statistics once. OPT-10 tests whether a better estimate (more independent checkpoints, and statistics re-estimated on
1,024 frames) improves E6 with the inference and post-processing unchanged.

## Frozen candidates (`movd.opt10.GRID`; 5 in total, no continuous parameter)

    c0 {"bn": true, "epoch_weights": [0.8, 0.2], "phase": 0, "runs": ["A"]}
    m2 {"bn": false, "epoch_weights": [0.8, 0.2], "phase": 1, "runs": ["A", "R1"]}
    m4 {"bn": false, "epoch_weights": [0.8, 0.2], "phase": 2, "runs": ["A", "R1", "R2", "R3"]}
    m4e1 {"bn": false, "epoch_weights": [1.0, 0.0], "phase": 2, "runs": ["A", "R1", "R2", "R3"]}
    m4bn {"bn": true, "epoch_weights": [0.8, 0.2], "phase": 2, "runs": ["A", "R1", "R2", "R3"]}

* `runs`: the fine-tuning runs averaged uniformly. `A` = Model A's run (CH-14 `final_rtdetr`). `R1`, `R2`, `R3` =
  seed replicates: `configs/runs/final_rtdetr.json` with `overrides.seed` 43 / 44 / 45 and `stop_after_epoch` 2
  (same 6-epoch schedule, `epoch0.pt` and `epoch1.pt` saved). Seeds 43-45 are the approved plan.md section 11
  exception to the frozen seed 42.
* `epoch_weights`: the weight of each run's epoch-1 and epoch-2 checkpoint; a run's checkpoint gets
  `epoch_weights[e - 1] / number of runs` (`movd.opt10.candidate_weights`). `[0.8, 0.2]` is E6's own formula; for
  run `A` alone it is soup w080. `m4e1` uses the epoch-1 checkpoints only.
* Arithmetic: `movd.opt10.weighted_state`, float32 in the listed order, stored in the checkpoints' float16, the
  OPT-08 soup arithmetic generalised to N checkpoints.
* `bn: true`: BatchNorm re-estimation (next section) after the averaging. `c0` = soup w080 with re-estimated
  statistics and **no** other change (all parameters identical to soup w080; checked at build time).
* `m4bn` is built only if `c0`'s discovery delta is >= 0. Phase 2 screens `m4`, `m4e1` (and `m4bn`) on discovery
  and selects the single best; `c0` (Phase 0) and `m2` (Phase 1) are judged alone.

## BatchNorm re-estimation (defined once, used by `c0` and `m4bn`)

* **Frames:** exactly 1,024 unique train_final frames at evenly spaced positions of the sorted train_final id list
  (6,072 ids): index `round(i * 6071 / 1023)` for i = 0 .. 1023 (`movd.opt10.spaced_indices`, integer arithmetic).
  First index 0, last index 6,071, index gaps 5 or 6, all 56 train clips covered (2 to 60 frames per clip). This is
  neither the first 1,024 frames nor `[::6]` (which gives 1,012). The ids are stored in `bn_frames.json`; sha256 of
  the comma-joined ids: `7a80e65ff02808e4a17e3bd6725fee3fff075907b38abf97698f75bc5ba008fe`.
* **Images:** the masked train_final export (`yolo_data_final/train_final/images/`, the files the trainer read;
  manifest equal to CH-14's), stretched to 1344 x 1344 (bilinear), RGB, / 255, as in training and prediction. No flip.
* **Procedure** (`movd.opt10.reestimate_bn`): the whole network in eval mode, then **only** the BatchNorm layers in
  training mode; each layer's `running_mean`, `running_var` and `num_batches_tracked` reset and `momentum=None`
  (plain cumulative average of the per-image statistics); batch 1, FP32, `torch.no_grad()`; afterwards every
  momentum (0.03) and every module's training flag restored. If a batch fails, the original statistics are put back.
  Only BatchNorm buffers change. The checkpoint is saved like a soup (float16, `ema` entry), and its sha256, the
  sha256 of the 1,024 image files and a parameter-identity check are recorded.
* The inference path of a re-estimated checkpoint is the unchanged E6 path (Ultralytics predictor, CH-13 views).

## Diagnostics D1 and D2 (Phase 0; reported, **never selected from**)

Single full-frame view at 1344, top-300 decoding, conf 0.001, project evaluator; checkpoints = epochs 1, 3, 6 of
`final_rtdetr` (`epoch0.pt`, `epoch2.pt`, `epoch5.pt`, hashes in `results/ch-14-final/checkpoints.json`).
The Ultralytics predictor folds BatchNorm into the convolutions, so D1 runs the unfused network directly
(`movd.opt10.direct_raw`).

* **Smoke / anchor (prerequisite of D1, D2 and `c0`):** the unfused network in normal eval mode must give Model A's
  stored val_fast mAP 0.27896670695801273 (`results/final_rtdetr/curve.json`, epoch 1) within +-0.0005, and the
  OPT-10 raw-cache path must reproduce 20 frames of OPT-08's protected w080 shard exactly. If either fails: stop,
  no diagnostic and no `c0`.
* **D1:** the three checkpoints on val_fast (714 frames) with BatchNorm using the statistics of the current image
  (`movd.opt10.bn_batch_stats`; the stored statistics must be byte-identical afterwards), each compared with the same
  checkpoint in normal eval mode through the same unfused path.
* **D2:** the three checkpoints in normal eval mode on 714 train_final frames (unmasked originals, train ground
  truth, same evenly spaced rule with 714 frames; ids in `d2_frames.json`, sha256 of the comma-joined ids
  `1d34971dcf2f2f68517ff08dec3041c64d7fc391f92ddfb7e1e900786fbef6fb`). Skipped, and recorded as skipped, if it
  cannot be run without editing `movd/evaluate.py`.

Interpretation table (fixed now). The candidate causes of the post-epoch-1 decline are H1 = batch-1 BatchNorm makes
training and inference disagree, H2 = the 33-frame statistics are a noisy sample, H3 = the model memorises the
6,072 correlated train frames.

| Observation | Reading |
|---|---|
| D1 lifts epoch 6 by >= 0.010 val_fast mAP over its eval-mode value | H1 is material; direction B (BatchNorm-consistent fine-tuning, a separate OPT-11) becomes eligible as the fallback |
| `c0` discovery delta >= +0.0010 | H2 is material |
| D2 mAP rises by >= 0.03 from epoch 1 to epoch 6 while val_fast falls | H3 confirmed |
| none of the three | the cause stays unknown; averaging is the remaining lever |

The readings are not exclusive. No diagnostic number selects, tunes or gates a candidate.

## Stages (the OPT-08 split, `results/opt-08-overnight/split.json`, reused unchanged)

Discovery = clips 339, 086, 137, 182 (1,335 frames); confirmation = 305, 117, 268 (4K) (1,511 frames).
Baseline for every stage = unchanged E6 on the same frames (`results/opt-09-e6-tuning/baseline/subsets.json`).

* **S1 (discovery):** delta vs E6 >= **+0.0010**. Phase 2: the best of `m4`, `m4e1`, `m4bn` by discovery mAP (ties:
  the earlier one) is the only one that continues.
* **S2 (confirmation):** the candidate once on the confirmation clips; passes if delta vs E6 >= 0.
* **S3 (val_full gate):** once, only after S1 and S2 pass.
* The discovery raw cache is the only GPU inference before S1; the confirmation cache is built only after S1
  passes; the test cache only for an upload-ready candidate (never in the Phase 0 session).
* A candidate that fails a stage stops. It is never retuned or re-screened.

## Promotion gate vs unchanged E6 (S3, once per candidate)

| Check | Rule |
|---|---|
| G1 | val_full mAP >= E6 + 0.0010 |
| G2 | width <= 1920 subset (clips 086, 182, 305, 339; test-like) mAP >= E6 + 0.0010 (OPT-09 used + 0.0005) |
| G3 | at least 5 of 7 clips with per-clip delta >= 0 |
| G4 | val_full without the 4K clip `uav0000268_05773_v` mAP >= E6 + 0.0010 |
| G5 | AP_small loss <= 0.002 |
| G6 | density >= 100 bin mAP loss <= 0.002 |
| G7 | leave-one-clip-out: for each of the 7 clips removed in turn, val_full delta > 0 |
| G8 | deterministic: a separate process rebuilds the val_full predictions byte for byte (and, before a CSV, the test predictions sha256) |
| G9 | evaluator, protocol configs, submission writer and validators unchanged vs `origin/develop` |

All 8 class results, size bins and density bins are reported. Passing G1-G9 makes a candidate **upload-ready**.

* **Selection tier:** an upload-ready CSV may replace E4 as a Kaggle final selection only if its val_full delta is
  >= **+0.0015** *and* its public score exceeds E6's 0.2947349001. E6 stays a final selection in every outcome.
  Nothing is selected from the public leaderboard alone.
* **Label T** (upload priority only): val_full >= 0.2999.

Where the thresholds come from: +0.0014 is the smallest val_full gain in this project that produced a same-sign
public gain (OPT-07 x4 vs Model A, public +0.0013), and 0.0013 is the smallest public difference resolved between
two of our submissions. The five changes inside the Model A family transferred at public / val ratios 0.9-1.7.
So +0.0010 is the floor for a probe and +0.0015 for a selection; 0.300 public needs +0.0053, i.e. about +0.0033
val_full at the median ratio 1.6 (label T).

## Stop rules and budgets

* **Barrier:** `m2` discovery delta < -0.0020 -> no R2 / R3; fallback decision.
* Phase 2: no candidate passes S1 -> NEGATIVE closeout, no CSV.
* Training: a non-finite loss, or a run projected beyond 70 min, stops the run (logs kept).
* GPU: one process at a time, `nvidia-smi` before every job. Total <= 435 min: Phase 0 <= 45, Phase 1 <= 85,
  Phase 2 <= 200, test inference <= 105. At most 3 training runs (R1-R3, 2 epochs each). At most 3 new CSVs.
  Every GPU job is logged in `gpu_log.json`.
* Nothing is uploaded by Claude. E6, E4, Model A, `epoch1.pt`, soup w080 and the OPT-08 raw caches are never written.
* Fallback if no candidate is upload-ready: if D1 was positive (table above), propose OPT-11 (direction B) for
  student approval with its own frozen rule; otherwise stop optimising and keep E6 + E4.

## Session scope (approved 2026-11-11)

This session runs **Phase 0 only**: the smoke / anchor, D1, D2 and `c0` through S1 and, only if it passes, S2 and
S3. No replicate training (R1-R3), no `m2` / `m4*`, no test inference, no CSV, no upload. The session stops after
reporting D1, D2 and the `c0` decision; replicate training needs a separate student approval.
