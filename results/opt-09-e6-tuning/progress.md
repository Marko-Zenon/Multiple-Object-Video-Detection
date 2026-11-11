# OPT-09 progress log

Focused local search around the protected E6 pipeline. Worker: Claude for Badabuba. Branch `opt-09-e6-tuning`,
worktree `C:/UCU/AI/hw2/MOVD-opt09`. Rules: [`selection_rule.md`](selection_rule.md). Times are Europe/Kyiv.

## 2026-11-11 10:30 — claim and start checks

* `origin/develop` = a848486 (PR #26 merged 2026-11-11 04:28 UTC); OPT-08 ✅ DONE; no open PR into develop;
  notebook lock FREE; RTX 5070 idle (1.9 GB used by the desktop, 0 %), no Python process.
* Claim pushed to develop (9516b2f). The row first said 08:30: Git Bash ignored `TZ=Europe/Kyiv` and printed UTC.
  The claim was at 10:30 Kyiv (commit time 10:29:54 +02:00). Budget: until 13:30 Kyiv.
* `check_dataset.py` PASS; `make_splits.py --check` PASS (val_fast / val_full hashes as committed).
* Kaggle public scores (user-reported) added to `results/submissions.csv`: E6 0.2947349001, E4 0.2913788894,
  x2 0.2805168313, x4 0.2787106940. E6 and E4 are the 2 current Kaggle final selections (user-reported).

## 10:40 — identity (`identity.json`)

All protected hashes match: Model A `6ff4a430…5b0b`, epoch1.pt `5594fb0e…5799`, soup w080 `2d60d3fa…d939`,
E6 val_full `bdb2806e…ad52`, E6 test `ef3f5c2f…0aa`, E6 CSV `35a225c9…7c35`, E4 CSV `7dc9b318…2014`,
x2 CSV `9dc8fb03…c867`, x4 CSV `09a5c8d1…2fcb`; 15 E6 raw shards hashed; frozen files equal `origin/develop`.
The OPT-09 code path (parallel merge + `opt09.five_frame_boost` at IoU 0.5, strength 1) rebuilds E6 val_full
byte-identically (367 s CPU).

## 10:50 — rule commit 431a13d, Phase B build + smoke, baseline

* `selection_rule.md`, `movd/opt09.py`, the scripts and tests committed and pushed (431a13d) before any evaluation.
* `opt09_soup.py build` (`b_soup_weight/soup_build.json`): OPT-08 compatibility passes (941 tensors, same names,
  shapes, dtypes, adjacent epochs); rebuilding w080 with the same arithmetic gives the E6 soup's tensors exactly;
  w075 `cfaaf0a4…b39d`, w085 `9353e0fd…1037`, w090 `ef65198e…a6f` (git-ignored `outputs/opt09/soup/`).
* `opt09_soup.py smoke` (`b_soup_weight/smoke.json`): 20 GPU frames of w080 equal OPT-08's raw shard exactly;
  w075 / w085 / w090 finite, scores in [0, 1], 0.46 s per frame. GPU checked idle before (2.0 GB desktop, 0 %).
* Discovery raw caches for w075, w085, w090 started in the background (one GPU process, sequential). No Phase B
  result is evaluated before Phase A is decided.
* Baseline (`baseline/subsets.json`): E6 evaluated once on every subset; equals the OPT-08 E6 record on val_full,
  width <= 1920, discovery (0.334488) and confirmation (0.270895).

## 11:00 — Phase A1 border margin: rejected at S1

| cand | margin | discovery mAP | Δ vs E6 | Δ AP_S |
|---|---:|---:|---:|---:|
| E6 | 0.005 | 0.334488 | — | — |
| m1 | 0.0035 | 0.334435 | -0.000053 | -0.000068 |
| m2 | 0.0065 | 0.334531 | +0.000043 | +0.000050 |
| m3 | 0.008 | 0.334533 | +0.000045 | +0.000023 |

Selection m3, +0.000045 < +0.0003: **rejected at S1**, no confirmation. The margin is flat around 0.005 (as in
OPT-03, where 0.005 and 0.01 differed by 0.000002 on val_fast). Not retuned.

## 11:08 — Phase A2 five-frame persistence: rejected at S1 (Phase A complete, no component passed)

| cand | change | discovery mAP | Δ vs E6 | Δ AP_S |
|---|---|---:|---:|---:|
| E6 | IoU 0.5, strength 1 | 0.334488 | — | — |
| p1 | strength 0.75 | 0.334069 | -0.000420 | -0.000263 |
| p2 | strength 1.25 | 0.334695 | +0.000207 | +0.000176 |
| p3 | strength 1.5 | 0.333311 | -0.001177 | -0.000714 |
| p4 | IoU 0.45 | 0.334169 | -0.000319 | -0.000362 |
| p5 | IoU 0.55 | 0.334496 | +0.000008 | +0.000084 |

Selection p2, +0.000207 < +0.0003: **rejected at S1**, no confirmation. Weaker (0.75) and much stronger (1.5)
boosts both lose; IoU 0.55 is flat and 0.45 loses. E6's IoU 0.5 / strength 1 sits at or near a local optimum.
Phase A result: neither border nor persistence passes; both stay as in E6. Not retuned.
