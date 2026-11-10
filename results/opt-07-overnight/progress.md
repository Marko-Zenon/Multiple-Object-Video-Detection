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
| 3 | x3_interpolation | [rule](x3_interpolation/rule.md) | pending |
| 4 | x4_border_confirmation | rule to be written and committed before its evaluation (needs Model A raw shards) | pending |
| 5 | x5_combination | defined below | pending |

**x5 (fixed now):** "independently positive" = a candidate among x1–x4 whose own val_full delta passes P1 (>= +0.0005)
and P3 and P4. If at least two exist, x5 applies all of them with their frozen parameters in list order (x4 first, on
the raw cache, if included; then x1 → x2 → x3, each on the previous output). If fewer than two exist, x5 is not run.

## Limits

CPU on cached predictions; GPU only to fill missing Model A raw shards (<= 90 min total, one process). No training,
no RT-DETR-x, no RT-DETR-l at 1600, no A + F ensemble, no evaluator / protocol change. Not touched: CH-14, CH-16,
report.md, homework.ipynb. Stop at: 2 upload-ready CSVs, all directions failed, 6 iterations, unsafe repo state, or
2026-11-11 08:00 Kyiv.

GPU minutes used: 0.

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

### Iteration 2 — 2026-11-11 00:39–00:50 Kyiv

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
