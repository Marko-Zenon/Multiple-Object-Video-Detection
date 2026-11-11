# OPT-08 — progress log

Read this file, [`TASK.md`](TASK.md) and `git log` at the start of every iteration. Times are Europe/Kyiv.

## Status

| # | Experiment | Rule commit | S1 discovery | S2 confirmation | S3 val_full / gate | CSV | Decision |
|---|---|---|---|---|---|---|---|
| 1 | e1_x4_then_x2 | 0e73d20 | c1 +0.0015 pass | +0.0011 pass | 0.2892 (+0.0014), U1-U8 pass | `17273d34…dbf8` | **UPLOAD-READY 1** |
| 2 | e2_tracklet_smoothing | 0e73d20 | — | — | — | — | pending |
| 3 | e3_broader_support | 0e73d20 | — | — | — | — | pending |
| 4 | e4_checkpoint_soup | rule pending | — | — | — | — | pending |
| 5 | e5_lowlr_epoch | decision pending | — | — | — | — | pending (optional) |
| 6 | e6_combination | TASK.md | — | — | — | — | pending |

Upload-ready CSVs: 1 of 4. GPU minutes used: 0 of 240.

## Iteration log

### Iteration 1 — 2026-11-11 01:55-

* Setup: PR #25 merged (2026-11-10 23:42 UTC), OPT-07 DONE on `origin/develop`, no previous loop state file, RTX 5070
  idle (0 %, desktop apps only), notebook lock FREE, no open PR to develop, no Python process. Claim b3965ed pushed.
* Worktree `MOVD-opt08` from `origin/develop`; dataset junction as in the other worktrees; Model A raw caches
  copied (not moved) from `MOVD-opt07/outputs/opt03/raw/` for the E1 x4 step.
* Frozen before any candidate result: split rule, stages S1-S3, gate U1-U8, E1-E3 grids, E4-E6 policy.
* Baseline: x2 on all 12 subsets reproduces the OPT-07 x2 record exactly (val_full 0.28786721769458246), 4.6 min.
* **E1 `e1_x4_then_x2` c1** (OPT-07 x4 then x2, unchanged): the x4 step reproduced the stored OPT-07 x4 predictions
  exactly on val_full and test. S1 discovery +0.001528 (pass); S2 confirmation
  +0.001135 (pass); S3 val_full 0.289248 vs x2 0.287867
  (+0.001381); width<=1920 +0.001092 (preferred +0.0005 met); AP_small +0.000347;
  >=100 +0.001822; 7/7 clips >= 0; without 4K +0.001320; U7 val_full sha identical in a
  fresh process, test sha identical across two processes; U8 unchanged. **Upload-ready CSV 1 of 4**:
  `C:/UCU/AI/hw2/MOVD-opt08/submissions/opt08_e1_x4_then_x2_c1_test.csv`, sha256 `17273d344bb0fed82804b69abba3d018ef89e8c04cc87291e78356f2b8f2dbf8`, 167,503,001 B, 3,811 rows,
  {'input_predictions': 1820997, 'accepted': 1820997, 'rejected_bad_score': 0, 'rejected_bad_box': 0, 'rejected_empty_box': 0, 'dropped_over_cap': 0, 'written': 1820997, 'rows': 3811, 'empty_rows': 0, 'max_dets': 500, 'float_format': 'repr (lossless)'} detections; project validator OK, independent validator valid (0 rows differ). Not uploaded. (02:17)
