# OPT-10 chunk 10.3c progress

## 2026-11-11T16:14:13+02:00 — new c0 evaluation continuation

Authorized by Badabuba from dec99da in the existing branch/worktree. Prior chunk 10.3's 45-minute guard and 41.235-minute accounting remain historical. Original D1/D2, c0 build, S1, discovery cache and raw predictions are read only. The original rule is unchanged.

This continuation has an independent 25-minute GPU budget for the existing c0 confirmation cache (13-minute admission estimate), with new timestamps and gpu_log.json in this directory. The budget was chosen before confirmation results. S2 uses the committed delta >= 0 rule. Only an S2 pass permits the committed S3 G1-G9 checks. No threshold changes. No training, test predictions, Kaggle CSV, upload, PR or DONE transition.

Fetched origin/develop; OPT-10 remains Badabuba's paused task. No other active task row/reservation or Python GPU worker found. NVIDIA shows desktop graphics use only. The new entry point reuses opt10_soup.cache("c0", "confirmation") and stores this continuation's accounting separately. opt10_loop.py changes only identity output locations so historical identity records remain unchanged. Forty OPT-10/accounting tests passed before GPU work. Protected and c0 identity verification is next; a fresh exclusive GPU heartbeat will be pushed after identity succeeds.

## 2026-11-11 16:23 Kyiv — identity passed, exclusive GPU heartbeat pushed

Continuation verification passed before CUDA. c0's authorized SHA256 matches, all historical files and discovery raw/prediction hashes match dec99da, protected E6/E4/Model A/epoch1/w080 hashes and all 55 protected raw shards match, and E6 full-validation predictions rebuilt byte-identically (358.5 seconds CPU). New verification records live only in this continuation directory; prior identity.json is unchanged.

Fetched origin/develop again. Badabuba still owns OPT-10; no other active board row/reservation or Python process found. Fresh exclusive GPU heartbeat e1a3646 pushed to origin/develop and merged into the existing branch before CUDA. The unchanged c0 confirmation cache procedure is running through the narrow continuation entry point, accounted only against this continuation's 25-minute budget. Historical gpu_log.json remains unchanged.

## 2026-11-11 16:46 Kyiv — S2 FAIL, c0 rejected; candidate evaluation stopped

The unchanged `opt10_loop.py confirm c0` evaluated the frozen 1,511 confirmation frames once. c0 mAP 0.2653289564689204 vs E6 0.27089469511565134, delta -0.005565738646730933 < 0: S2 FAIL. Original c0 S1 remains PASS (+0.001508015353369807). No S3, per-clip/leave-one-out full evaluation or determinism job is started after this rejection.

Confirmation AP_small is 0.012531503582036897 vs 0.012513525612396713 (slightly up); AP_large is 0.43153433415596437 vs 0.4397311578987566 (down). These are confirmation-only metrics and do not stand in for val_full G5 or other S3 gates. No c0 val_full metrics exist. Frozen thresholds remain unchanged.

Continuation GPU use: 885.4 seconds = 14.756667 minutes of its independent 25-minute cap, one cache job (three clips / 1,511 frames), peak reserved 0.86 GiB. Original chunk 10.3 GPU log and D1/D2/c0/S1 evidence remain unchanged. Only integrity and requested QA/closeout continue after S2 failure.


## 2026-11-11T16:58:27+02:00 — validation-only closeout complete

Full suite: 348 passed, 4 skipped (CUDA-only/missing reference weights). Dataset, all frozen split checks, scoped Ruff and format PASS. Final continuation identity verify PASS: all protected hashes, 55 raw shards and frozen files unchanged; E6 full reconstruction byte-identical (391.8 seconds); c0 and historical D1/D2/build/S1/discovery/log/decision bytes match dec99da and the start snapshot exactly. New confirmation shards and prediction hashes match their records.

Terminal decision remains REJECTED_AT_S2. S1 historical PASS, S2 FAIL, G9 integrity PASS; S3/G1-G8 NOT RUN because S2 failed. No c0 full-validation, test-like, dense, clip/leave-one-out or determinism result was invented. GPU reservation release follows the evidence push.
