# OPT-10 — task contract

Requested and approved by Badabuba (Nazar) on 2026-11-11. Worker: Claude. Branch `opt-10-soup-bn`, worktree
`C:/UCU/AI/hw2/MOVD-opt10`, board row OPT-10 (claim `df0ac04` on `origin/develop`, 2026-11-11 12:19 Kyiv).
Rules, grid, gates, diagnostics and budgets: [`selection_rule.md`](selection_rule.md), frozen before any result.

## Chunks

| # | Chunk | Session | State |
|---|---|---|---|
| 10.1 | Claim, worktree, frozen rule, frame lists, identity check (hashes, E6 val_full rebuilt byte-identically) | 1 (2026-11-11) | complete; start/end identity passed in continuation |
| 10.2 | Code + tests: N-checkpoint weighted average, BatchNorm re-estimation, per-image-statistics context, CLIs, replicate configs | 1 | complete; two validity-check corrections recorded in README.md |
| 10.3 | Phase 0 on the GPU (<= 45 min): smoke / anchor, D1, D2, `c0` through S1 (S2 / S3 only if S1 passes) | 1 | partial; D1/D2/c0/S1 complete, S2/S3 stopped by frozen GPU cap; **student review** |
| 10.4 | Phase 1: replicate smoke, R1 (seed 43, 2 epochs), single-view val_fast | later | needs a separate approval |
| 10.5 | `m2`: build, discovery cache, S1; barrier rule | later | needs approval |
| 10.6 | Phase 2: R2, R3 | later | needs approval |
| 10.7 | `m4`, `m4e1` (+ `m4bn`): S1 selection, S2, S3 | later | needs approval |
| 10.8 | Test cache, CSV, validators, determinism for upload-ready candidates (at most 3) | later | needs approval |
| 10.9 | Closeout, PR | later | — |

## Fixed for session 1

* Chunks 10.1-10.3 only. No training, no chunk 10.4 or later.
* No test inference, no CSV, no Kaggle upload.
* E6 and E4 artifacts, Model A, `epoch1.pt`, soup w080 and the OPT-08 raw caches are read only and hash-checked at
  the start and the end.
* One GPU process at a time; `nvidia-smi` before every job; Phase 0 GPU cap 45 min (`gpu_log.json`).
* Stop after reporting D1, D2 and the `c0` decision.

## Protocol exception

Replicate runs R1-R3 use seeds 43, 44, 45 through `overrides.seed` (plan.md section 11, approved by Badabuba on
2026-11-11). Not used in session 1.

## Continuation checkpoint

Codex resumed the latest pushed Fable checkpoint cffa98e on 2026-11-11 under the existing Badabuba ownership. See progress.md, README.md and session_decision.json. No new branch, worktree, claim or experiment was created. Chunk 10.3 is partial because confirmation admission exceeds the frozen Phase 0 GPU cap after S1 passed.

## Separately authorized chunk 10.3c — c0 evaluation continuation

Authorized by Badabuba on 2026-11-11 from dec99da. The previous chunk 10.3 and its 45-minute guard are retained as the completed historical execution checkpoint. This is a new continuation of only the pending c0 evaluation, with an independent 25-minute GPU budget and timestamps in continuation-c0-eval/contract.json and gpu_log.json. It does not restart Phase 0 or authorize chunk 10.4. Use the original selection_rule.md for S2, conditional S3, all promotion checks and determinism. Stop after the c0 decision. See continuation-c0-eval/progress.md.

Chunk 10.3c outcome: c0 REJECTED at S2 under the frozen delta >=0 rule. This authorized evaluation continuation is complete at its terminal rejection. Historical 10.3 records remain intact. OPT-10 itself remains PARTIAL / PAUSED for student review, with no later chunk or PR authorized.
