# OPT-07 — overnight post-processing loop on protected Model A (progress log)

Worker: Claude (autonomous loop, max 6 iterations), for Badabuba. Branch `opt-07-overnight`, worktree `MOVD-opt07`.
Claim: bf144ad on `origin/develop` (2026-11-11 00:25 Kyiv). Read this file and `git log` at the start of every iteration.

## Protected (never modified, re-hashed by `scripts/opt07_loop.py verify`)

| Item | Value |
|---|---|
| Model A checkpoint | `MOVD-ch14/outputs/final_rtdetr/train/weights/epoch0.pt`, sha256 `6ff4a430188405b5d13356872c63f141c4d3dbc7b747983cdf733ce386aa5b0b` |
| Model A CSV | `MOVD-ch14b/submissions/final_rtdetr_epoch1_ch13_fliptile_nms05_test.csv`, sha256 `e587c68b517d6d59f90122946d8a45ef5beac4fb8ae3b24f66ced1268def0da7` |
| Model A public score | 0.2774301362 (user-reported, `results/submissions.csv`) |
| Model A val_full | 0.2859619 (CH-14b registered) |

## Context that sets the bar

OPT-06A c4 (A x 1.0 + F x 0.75, NMS 0.6) raised val_full to 0.2886 (+0.0026) but scored **0.2695252954** public
(user-reported, 2026-11-11), below Model A's 0.2774301362. **A val_full gain alone is not sufficient.** The OPT-07 gate
therefore adds a test-like (width <= 1920) subset, a per-clip consistency check and a without-4K-clip check, and every
candidate here changes Model A's own predictions only (no second model).

## Promotion gate (fixed before the first candidate; deltas vs Model A on val_full, unrounded)

| ID | Criterion |
|---|---|
| P1 | val_full mAP >= A + 0.0005 |
| P2 | width <= 1920 subset (clips 086, 182, 305, 339; 1,286 frames) mAP >= A + 0.0005 |
| P3 | AP_small loss <= 0.003 |
| P4 | density >= 100 bin mAP loss <= 0.005 |
| P5 | at least 4 of the 7 val clips with per-clip mAP delta >= 0 |
| P6 | no dependence on the 4K clip only: val_full without `uav0000268_05773_v` (1,868 frames) mAP >= A + 0.0005 |
| P7 | deterministic: a fresh process reproduces the val_full predictions byte for byte (sha256), and the test predictions sha256 is identical across two processes |

Any failed criterion rejects the candidate. A promoted candidate gets one lossless test CSV (project + independent
validators; path, sha256 and rule recorded); nothing is uploaded. At most 2 upload-ready CSVs.

## Experiment order (one per iteration; parameters frozen in `movd/opt07.py` EXPERIMENTS before any evaluation)

| # | ID | Rule | Status |
|---|---|---|---|
| 1 | x1_isolation_demotion | [rule](x1_isolation_demotion/rule.md) | **rejected** (P1), not independently positive |
| 2 | x2_persistence_boost | [rule](x2_persistence_boost/rule.md) | **PROMOTED** (P1–P7), upload-ready CSV 1 of 2 |
| 3 | x3_interpolation | [rule](x3_interpolation/rule.md) | **rejected** (P1, P2, P6), not independently positive |
| 4 | x4_border_confirmation | [rule](x4_border_confirmation/rule.md) (6101fc7) | **PROMOTED** (P1–P7), upload-ready CSV 2 of 2 |
| 5 | x5_combination | defined below | **not run**: stop condition reached (2 upload-ready CSVs) |

**x5 (fixed now):** "independently positive" = a candidate among x1–x4 whose own val_full delta passes P1 (>= +0.0005)
and P3 and P4. If at least two exist, x5 applies all of them with their frozen parameters in list order (x4 first, on
the raw cache, if included; then x1 → x2 → x3, each on the previous output). If fewer than two exist, x5 is not run.

## Limits

CPU on cached predictions; GPU only to fill missing Model A raw shards (<= 90 min total, one process). No training,
no RT-DETR-x, no RT-DETR-l at 1600, no A + F ensemble, no evaluator / protocol change. Not touched: CH-14, CH-16,
report.md, homework.ipynb. Stop at: 2 upload-ready CSVs, all directions failed, 6 iterations, unsafe repo state, or
2026-11-11 08:00 Kyiv.

GPU minutes used: 40.4 of 90 (val_full shards 11.7 min, test cache 28.7 min; Model A raw shards only; one process at a time). RTX 5070 released.

## Iteration log

### Iteration 1 — 2026-11-11 00:25–00:38 Kyiv

* Setup: claim bf144ad; worktree `MOVD-opt07` (dataset via junction); `verify` ok (all 4 protected hashes match,
  `identity.json`); rules x1–x3, gate and transforms committed in **771b851 before any result**, pushed.
* Baseline (`baseline/subsets.json`): Model A val_full 0.28596193196293734 = registered CH-14b value exactly; 4.1 min.
* **x1_isolation_demotion — REJECTED (P1).** 178,041 of 1,405,788 scores halved. val_full +0.00020 (needs +0.0005);
  width<=1920 +0.00149; without-4K +0.00060; AP_small -0.00161; >=100 bin -0.00219; clips non-negative 4/7
  (137: -0.00317, 182: -0.00125, 086: -0.00009; 305: +0.00497). P2–P6 pass, P1 fails. Not "independently positive"
  (P1). Evidence `x1_isolation_demotion/` (gate.json, subsets.csv, per_class.csv, val_full/). No determinism step,
  no CSV.
* Next iteration: x2_persistence_boost (rule already frozen in 771b851).

### Iteration 2 — 2026-11-11 00:39–00:45 Kyiv

* `verify` ok. Rule frozen in 771b851 (before any result); parameters `{"iou": 0.5, "kind": "persistence_boost"}`.
* **x2_persistence_boost — PROMOTED (P1–P7 all pass).** 395,849 of 1,405,788 scores raised (never lowered, no box
  added or removed). val_full 0.28596 -> **0.28787 (+0.00191)**, AP50 0.5390; width<=1920 **+0.00261**; without-4K
  **+0.00192**; AP_small **+0.00188**; >=100 bin **+0.00158**; clips non-negative **7/7** (086 +0.00066, 117 +0.00579,
  137 +0.00138, 182 +0.00136, 268 (4K) +0.00299, 305 +0.00412, 339 +0.00234); all 8 classes up (+0.0003 bus …
  +0.0038 tricycle). Transform 9.7 s; evaluation 3.9 min.
* P7: fresh-process val_full re-run sha256 identical (`6c3a929f…2c03`); test predictions identical across two
  processes (`fc4fdc8f…9191`) (`determinism.json`).
* **Upload-ready CSV 1/2** (not uploaded): `C:/UCU/AI/hw2/MOVD-opt07/submissions/opt07_x2_persistence_boost_test.csv`,
  172,652,039 bytes, sha256 `9dc8fb03dc708b2b78b05afeff29289317a810f0219e9be4e2fd50029433c867`, 3,811 rows, 1,875,374
  detections (= Model A count; scores only), lossless repr floats; project validator OK, independent validator
  valid (0 rows differ from the predictions) (`x2_persistence_boost/test/`).
* x2 counts as "independently positive" for x5.
* Next iteration: x3_interpolation (rule frozen in 771b851).

### Iteration 3 — 2026-11-11 00:46–00:52 Kyiv

* `verify` ok. Rule frozen in 771b851 (before any result).
* **x3_interpolation — REJECTED (P1, P2, P6).** Net +25 boxes (1,405,788 -> 1,405,813): 2,368 of 2,846 frames are
  already at the 500 cap, so added midpoint boxes mostly displace the lowest-score boxes. val_full +0.00005;
  width<=1920 +0.00001; without-4K +0.00004; AP_small +0.00004; >=100 bin +0.00014; clips non-negative 5/7
  (086 -0.000001, 182 -0.000005). Effect is negligible. Not "independently positive" (P1). No determinism step, no CSV.
* Independently positive so far: x2 only. x5 needs at least two, so it depends on x4.
* Next iteration: x4_border_confirmation (rule to be written and committed first; needs Model A raw shards).

### Iterations 4–5 — 2026-11-11 00:50–01:40 Kyiv (x4; loop iteration 4 ended while the GPU fill ran, iteration 5 finished x4)

* x4 = OPT-03 e2 unchanged (drop tile boxes within 0.005 of an inner tile edge before the CH-13 merge), never
  evaluated on val_full before. Rule + code + 2 tests committed in **6101fc7 before the cache fill and any x4 result**.
* Raw cache: 4 OPT-03 val_full shards copied (sha256 identical); 3 missing clips (268, 305, 339; 1,437 frames) filled
  by the unchanged `scripts/opt03_cache.py` (checkpoint sha256 checked), 11.7 GPU min, 0.49 s/frame, peak 0.70 GB
  (`cache_val_full.json`). **Identity:** cache merged at factor 1 == Model A stored val_full predictions, 1,405,788
  boxes, max abs diff 0.0 (`identity_val_full.json`).
* **x4_border_confirmation — PROMOTED (P1–P7 all pass).** val_full 0.28596 -> **0.28739 (+0.00142)**, AP50 0.5382;
  width<=1920 **+0.00109**; without-4K **+0.00136**; AP_small **+0.00036**; >=100 bin **+0.00186**; clips
  non-negative **7/7** (086 +0.00015, 117 +0.00261, 137 +0.00159, 182 +0.00116, 268 (4K) +0.00141, 305 +0.00085,
  339 +0.00129); all 8 classes up (+0.0003 truck … +0.0029 tricycle). 1,384,308 val_full boxes (fewer than A).
* Test raw cache 3,811 frames, 28.7 GPU min (`cache_test.json`); **test identity** at factor 1 == Model A stored test
  predictions, 1,875,374 boxes, max abs diff 0.0 (`identity_test.json`). P7: val_full re-run sha256 identical
  (`a7c97b07…1a5f`), test identical across two processes (`bb7adfaa…6b8c`).
* **Upload-ready CSV 2/2** (not uploaded): `C:/UCU/AI/hw2/MOVD-opt07/submissions/opt07_x4_border_confirmation_test.csv`,
  167,530,212 bytes, sha256 `09a5c8d14633f4bade9b9c5b2edd6122c38ca8c3de0cf8afa6b5f79184cd2fcb`, 3,811 rows,
  1,820,997 detections, lossless; project validator OK, independent validator valid (`x4_border_confirmation/test/`).

## Stop

Terminal condition reached at 2026-11-11 01:40 Kyiv: **two upload-ready CSVs exist** (x2, x4). x5 (x2 on top of x4,
both independently positive) was **not run** because the CSV cap is reached; it stays an untested idea. Final
summary and recommended upload order: [`README.md`](README.md).
