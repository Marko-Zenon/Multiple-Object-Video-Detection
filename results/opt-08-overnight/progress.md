# OPT-08 — progress log

Read this file, [`TASK.md`](TASK.md) and `git log` at the start of every iteration. Times are Europe/Kyiv.

## Status

| # | Experiment | Rule commit | S1 discovery | S2 confirmation | S3 val_full / gate | CSV | Decision |
|---|---|---|---|---|---|---|---|
| 1 | e1_x4_then_x2 | (this commit) | — | — | — | — | pending |
| 2 | e2_tracklet_smoothing | (this commit) | — | — | — | — | pending |
| 3 | e3_broader_support | (this commit) | — | — | — | — | pending |
| 4 | e4_checkpoint_soup | rule pending | — | — | — | — | pending |
| 5 | e5_lowlr_epoch | decision pending | — | — | — | — | pending (optional) |
| 6 | e6_combination | TASK.md | — | — | — | — | pending |

Upload-ready CSVs: 0 of 4. GPU minutes used: 0 of 240.

## Iteration log

### Iteration 1 — 2026-11-11 01:55-

* Setup: PR #25 merged (2026-11-10 23:42 UTC), OPT-07 DONE on `origin/develop`, no previous loop state file, RTX 5070
  idle (0 %, desktop apps only), notebook lock FREE, no open PR to develop, no Python process. Claim b3965ed pushed.
* Worktree `MOVD-opt08` from `origin/develop`; dataset junction as in the other worktrees; Model A raw caches
  copied (not moved) from `MOVD-opt07/outputs/opt03/raw/` for the E1 x4 step.
* Frozen before any candidate result: split rule, stages S1-S3, gate U1-U8, E1-E3 grids, E4-E6 policy.
