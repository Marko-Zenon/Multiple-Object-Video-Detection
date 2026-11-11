# OPT-10 progress

## 2026-11-11 13:56 Kyiv — resumed at pushed checkpoint cffa98e

Badabuba authorized Codex to continue Fable's chunks 10.1/10.2 in the existing `opt-10-soup-bn` branch and `MOVD-opt10` worktree. No earlier Phase 0 outputs existed. The frozen rule, frame lists, implementation and git history were read before execution.

- Fetched origin. OPT-10 remains IN_PROGRESS under Badabuba. Task branch matched its pushed checkpoint cffa98e.
- No conflicting Python process or NVIDIA compute process at preflight. Desktop VRAM use approximately 2 GiB.
- Exclusive RTX 5070 GPU reservation pushed to origin/develop in heartbeat 4e93f72 and merged into the existing task branch (6380501) before CUDA.
- Session limited to identity checks and chunk 10.3. Phase 0 GPU budget is 45 minutes. Stop for student review after D1, D2 and c0's frozen stage decision.
- Python runtime uses uv CPython 3.12 with the repository's existing site-packages via PYTHONPATH. Torch 2.14.1+cu130, Ultralytics 8.4.207. CPU commands hide CUDA.
- Initial `opt10_loop.py verify` running on CPU, followed by `opt10_soup.py check`. Logs in ignored outputs/opt10/logs.

## 2026-11-11 14:03 Kyiv — identity and raw-cache smoke passed

- `opt10_loop.py verify` PASS: all protected hashes and 55 raw shards match, frozen files equal origin/develop, E6 full-validation prediction rebuild byte-identical (bdb2806e...cad52), generalized soup tensors equal w080. CPU rebuild 366.6 seconds.
- `opt10_soup.py check` PASS: all 941 tensors identical to E6 soup w080 using the frozen 0.8/0.2 average.
- `opt10_soup.py smoke` PASS: 20 frames / 6 views reproduce OPT-08's protected raw shard exactly. Inference 12.7 seconds, peak reserved 0.86 GiB. No conflicting Python job before smoke or D1.
- D1 started after these prerequisites, with its unfused normal-eval anchor checked before per-image-statistics inference.

## 2026-11-11 14:19 Kyiv — D1 complete, H1 not supported

| epoch | stored fused val_fast mAP | unfused eval mAP | per-image BN mAP | lift |
|---|---:|---:|---:|---:|
| 1 | 0.278966707 | 0.278989974 | 0.256616625 | -0.022373349 |
| 3 | 0.261371116 | 0.261349820 | 0.243914599 | -0.017435221 |
| 6 | 0.238126383 | 0.238084752 | 0.233885170 | -0.004199582 |

Anchor passed (difference 0.000023267 vs tolerance 0.0005). State restored byte-identically at all three epochs. The frozen H1 criterion is epoch-6 lift >= +0.010; observed -0.004199582, so H1 is not supported. Full six-metric rows are in diagnostics/d1.json. No diagnostic is used to select a checkpoint. D2 started after another process/GPU preflight.

## 2026-11-11 14:28 Kyiv — D2 complete, H3 confirmed

| epoch | train-frame mAP | AP50 | AP75 | AP_small | AP_medium | AP_large |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.397074777 | 0.692873962 | 0.399839886 | 0.249959559 | 0.390724972 | 0.580053381 |
| 3 | 0.477392397 | 0.801686188 | 0.494474800 | 0.323713383 | 0.483948795 | 0.636769752 |
| 6 | 0.499405821 | 0.825606311 | 0.523386757 | 0.338506376 | 0.505452235 | 0.654685267 |

Frozen 714-frame list hash matches; unmasked train originals, normal eval mode, unchanged evaluator. Epoch 1 -> 6 train mAP +0.102331044 >= +0.03 while normal-eval validation mAP falls -0.040905222. H3 confirmed under the frozen reading. GPU-accounted wall time after D2 is 23.9 min.

## 2026-11-11 14:29 Kyiv — existing manifest lookup corrected before c0 CUDA

The first `bn c0` attempt stopped before the GPU-job context because the CLI referred to a nonexistent tracked `results/ch-14-final/export/train_final_manifest.json`. CH-14 stores the same reference manifest inside `results/ch-14-final/export/export_check.json`. Corrected only this reference lookup and compared the actual masked export manifest to that committed embedded object. Regression test checks acceptance and rejection of a changed manifest. Scoped OPT-10 tests: 36 passed. Fix committed as 47e0f39 before building c0. The frozen rule, frame lists, averaging, BN algorithm and protected artifacts were unchanged. Failed-attempt log retained in ignored outputs/opt10/logs/bn_c0.txt. c0 build resumed after another GPU/process preflight.

## 2026-11-11 14:35 Kyiv — c0 built, valid zero-variance channels audited

The frozen re-estimation completed once: 1,024 batches, 122 BN layers, every counter 1,024; non-BN tensors and saved parameters identical to w080; flags and momenta restored. Checkpoint SHA256 6acb20696921a1b5e1fd7b63d0d17d76f6349d7a1f1eba785f0d0f441a16002c. Image-content SHA256 b8c9bb9d89a7b5266608df63dc2b1fa750d0f19c40f8482c327adc0a5b1bf69f. GPU-accounted re-estimation wall time 208.4 seconds.

The original build check falsely required strictly positive variance. Eight constant channels have zero variance (7 in model.6.ec.bn and 1 in model.7.ec.bn), exactly as in the protected w080 checkpoint, with epsilon 0.001 and no negative variance. The saved checkpoint is finite. Corrected this implementation validity check to finite nonnegative variance (ebb72d8), added a constant-channel regression test (37 OPT-10 tests pass), retained build_FAILED.json and variance_audit.json, then CPU-audited the same completed checkpoint and wrote build.json. No tensor changed, no BN rerun, no candidate retuning, no selection threshold changed. Median relative BN mean / variance changes are 0.003793946 / 0.010448006. Discovery inference is next.


## 2026-11-11 15:04 Kyiv — S1 passed, frozen budget stops confirmation; validation complete

- Discovery cache: 1,335 frames, four clips, 828.8 seconds inference; S1 CPU wall 202.6 seconds.
- c0 discovery mAP 0.335996060382765 vs E6 0.3344880450293952, delta +0.001508015353369807 >= +0.0010: S1 PASS. H2 material by the frozen reading; m4bn eligible under its discovery condition, with later chunks still requiring approval.
- Discovery AP_small 0.19168673953689122 vs E6 0.19486829336528577, delta -0.003181553828394551. Reported; this is not a val_full G5 evaluation. Full six-metric tables in README.md and c0/screen.json.
- Actual logged Phase 0 use 2474.1 seconds = 41.235 minutes <= 45. Unchanged budget guard refuses confirmation: used + 13-minute reservation = 54.235 > 45. Refusal happened before CUDA/cache creation; confirmation_budget_stop.txt retained. S2 and S3 not run; c0 is not upload-ready and has no val_full number. Chunk 10.3 remains partial at this explicit frozen budget stop.
- Full suite 345 passed / 4 skipped, with CUDA hidden for CPU checks; reasons in checks/pytest.txt. Full dataset and all four frozen split checks PASS. Scoped Ruff check and format PASS.
- End identity verify PASS: all protected hashes and all 55 raw shards match, E6 full predictions rebuilt byte-identically, weighted-state identity exact, frozen files unchanged. Start and end identity records both PASS.
- Stop for student review. Existing E6/E4 final selections protected. No chunk 10.4, replicate training, confirmation/test inference, submission or upload. GPU reservation release follows in the closing live-board heartbeat.


## 2026-11-11 15:06 Kyiv — GPU released, handoff pushed

Exclusive GPU reservation released on origin/develop in dbef9fd after the evidence checkpoint fca1b76 was pushed. No Python job remained at release; NVIDIA reported 1% utilization and 1,957 MiB desktop memory use. The coordination commit was merged into the existing task branch and pushed (d3528d0). OPT-10 stays PARTIAL / PAUSED under Badabuba for student review at the frozen budget stop. Stop here.


## 2026-11-11T16:58:27+02:00 — separately authorized 10.3c continuation, c0 rejected at S2

Authorized from dec99da with a new 25-minute GPU budget; previous chunk 10.3 and all original evidence/accounting retained unchanged. Confirmation c0 mAP 0.2653289564689204 vs E6 0.27089469511565134, delta -0.005565738646730933 < 0: frozen S2 FAIL. Candidate evaluation stopped; S3 and G1-G8 are unreached, G9/start-end integrity PASS. Independent GPU use 14.756667 minutes, one confirmation cache job, no Phase 0 restart. Full suite 348 passed / 4 skipped; dataset/splits/scoped Ruff PASS. Detailed metrics and every gate status are in continuation-c0-eval/README.md and decision.json. Stop after c0 decision; GPU release recorded in closing heartbeat.

Continuation 10.3c GPU released at 2026-11-11T16:59:45+02:00, live-board commit 680a0117dcdfb94a211ca10a9992a22d417874a7. Terminal c0 S2 rejection is recorded; OPT-10 remains paused for student review.


## 2026-11-12 01:40 Kyiv — chunk 10.4 + 10.5 (R1, m2): m2 rejected at S1, barrier triggered

Authorized by Badabuba from 4066c4c; contract in phase1-r1/contract.json; board heartbeat a3229e3 reserved the RTX 5070. Start identity verify PASS.
- Replicate smoke PASS (seed 43 in args.yaml, stop after epoch 1 + resume, finite losses, inference finite, 3.98 GB).
- R1 trained once with configs/runs/opt10_rep_s43.json: only `seed` differs from Model A's kwargs, start weights 6de60b10…e73f, train_final 6,072 frames, 2 of 6 scheduled epochs, 12,144 images seen, finite losses, 52.9 min, 3.46 GB peak reserved. epoch0.pt 0e17db9e…62c9, epoch1.pt 5c02bd24…cee8.
- Fixed one implementation defect before the build: replicate checkpoint paths are relative to the hw2 folder (run_checkpoint joined them to the repo root); regression test added.
- m2 built exactly (0.4/0.4/0.1/0.1), sha256 7392eab9…c91c. Discovery cache 614 s.
- S1: m2 discovery mAP 0.325862930885 vs E6 0.334488045029, delta -0.008624870045 < +0.0010: REJECTED. Below the -0.0020 barrier: no R2/R3. S2/S3/G1-G9, test inference and CSV not run.
- R1 alone on val_fast (reported only): epoch 1 0.249933 (Model A 0.278967), epoch 2 0.250523 (Model A 0.258465).
- Phase 1 GPU 69.8 of 85 min. End identity verify PASS; 351 passed / 2 skipped; dataset, splits, scoped Ruff PASS. Details: phase1-r1/README.md, decision.json.
