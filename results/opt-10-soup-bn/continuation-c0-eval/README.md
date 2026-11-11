# OPT-10 chunk 10.3c — c0 rejected at frozen S2

New continuation authorized by Badabuba from dec99da on 2026-11-11, using the existing opt-10-soup-bn branch and MOVD-opt10 worktree. Original chunk 10.3, its 45-minute guard, D1/D2, c0 build and S1 evidence are historical and preserved. This continuation uses a separately committed 25-minute GPU budget and timestamps.

## Frozen stage decision

| check | required | observed | result |
|---|---|---|---|
| S1, historical discovery | delta mAP >= +0.0010 | +0.001508015353369807 | PASS, unchanged |
| S2 confirmation | delta mAP >= 0 | -0.005565738646730933 | FAIL |
| S3 promotion stage | S2 must pass | S2 failed | NOT RUN |
| G1 overall val_full | delta mAP >= +0.0010 | S3 not reached | NOT RUN |
| G2 width <=1920 | delta mAP >= +0.0010 | S3 not reached | NOT RUN |
| G3 clip consistency | at least 5/7 clips non-negative | S3 not reached | NOT RUN |
| G4 without 4K | delta mAP >= +0.0010 | S3 not reached | NOT RUN |
| G5 val_full AP_small | loss <= 0.002 | S3 not reached | NOT RUN |
| G6 dense >=100 | loss <= 0.002 | S3 not reached | NOT RUN |
| G7 leave-one-out | every delta > 0 | S3 not reached | NOT RUN |
| G8 determinism | separate-process val_full byte identity | S3 not reached | NOT RUN |
| G9 frozen evaluator/protocol/writer/validators | unchanged vs origin/develop | identity verifier passed | PASS |

The S2 failure is the terminal c0 decision under the original selection_rule.md. No candidate threshold was changed, no S1 was repeated, and no S3 or extra candidate evaluation was started after failure. No c0 val_full score or promotion is claimed.

## Confirmation metrics — 1,511 frames

| metric | E6 | c0 | delta |
|---|---:|---:|---:|
| mAP | 0.270894695116 | 0.265328956469 | -0.005565738647 |
| AP50 | 0.449209740887 | 0.442029414955 | -0.007180325932 |
| AP75 | 0.285175737752 | 0.276761591082 | -0.008414146671 |
| AP_small | 0.012513525612 | 0.012531503582 | +0.000017977970 |
| AP_medium | 0.195383862370 | 0.195340686778 | -0.000043175592 |
| AP_large | 0.439731157899 | 0.431534334156 | -0.008196823743 |

AP_small is almost flat on confirmation, while AP_large falls by 0.008196823743. These confirmation metrics do not replace val_full G5 or a full-validation large-object result. Dense-frame, non-4K/test-like, per-clip and leave-one-out c0 metrics are S3 outputs and remain unreached under the frozen stop rule. G8 is also unreached because there is no c0 val_full promotion result to verify.

## Continuation budget and integrity

One GPU job cached the three frozen confirmation clips / 1,511 frames. Logged GPU-accounted wall time 885.4 seconds = 14.756666667 minutes <= independent 25-minute cap. Peak reserved memory 0.86 GiB. Prior Phase 0 gpu_log.json is unchanged at 41.235 of 45 minutes.

c0 checkpoint sha256 is 6acb20696921a1b5e1fd7b63d0d17d76f6349d7a1f1eba785f0d0f441a16002c. Start verification checked the protected E6/E4/Model A/epoch1/w080 artifacts and all 55 protected raw shards, historical Phase 0 evidence, old c0 discovery shards and predictions, and rebuilt E6 full predictions byte-identically. Start and final verifications both passed in this continuation directory. Final E6 reconstruction took 391.8 seconds and was byte-identical; all protected/historical artifact bytes match again.

Exclusive GPU heartbeat e1a3646 was pushed to origin/develop after identity and availability checks, before CUDA. The narrow continuation entry point reuses opt10_soup.cache for c0 confirmation, with separate accounting. The two opt10_loop.py edits only redirect verification evidence/output locations for this continuation. S2 uses the original confirm command, without scoring or gate changes.

## QA and preserved evidence

- Full suite 348 passed, 4 skipped (CUDA-only/missing reference-weight fixtures), with CUDA hidden for CPU QA. See checks/pytest.txt.
- Full dataset and frozen split checks passed. Scoped Ruff and format passed on six OPT-10 files.
- New identity records and GPU log are under continuation-c0-eval/. Original phase records remain intact.
- Confirmation metadata is ../c0/cache_confirmation.json and the one S2 evaluation is ../c0/confirm.json. Raw caches and confirmation predictions are ignored in outputs/opt10/.
- No R1/R2/R3 training, test predictions, Kaggle CSV, upload or PR. OPT-10 stays pending student review, never DONE.

Technical evidence only; student report prose and conclusions remain with Nazar and Marko.
