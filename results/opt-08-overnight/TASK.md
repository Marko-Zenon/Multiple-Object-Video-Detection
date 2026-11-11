# OPT-08 — bounded overnight optimization loop on the OPT-07 x2 baseline (task contract)

Requested by Badabuba (Nazar) on 2026-11-11 ~01:55 Kyiv as an autonomous loop (`/ralph-loop`, max 20 iterations,
completion promise `OPT08_DONE`). Worker: Claude. Branch `opt-08-overnight`, worktree `C:/UCU/AI/hw2/MOVD-opt08`,
board row OPT-08 (claim b3965ed on `origin/develop`, 2026-11-11 02:05 Kyiv). This file is fixed before the first
candidate result; later additions are appended with a timestamp and never change an earlier rule.

## Confirmed public results (user-reported)

| Submission | Public score | CSV sha256 |
|---|---:|---|
| Original Model A (CH-14b epoch 1 + CH-13 pipeline) | 0.2774301362 | `e587c68b…0da7` |
| **OPT-07 x2 persistence boost — protected baseline** | **0.2805168313** | `9dc8fb03dc708b2b78b05afeff29289317a810f0219e9be4e2fd50029433c867` |
| OPT-07 x4 border filtering | 0.2787106940 | `09a5c8d14633f4bade9b9c5b2edd6122c38ca8c3de0cf8afa6b5f79184cd2fcb` |
| OPT-06A c4 ensemble | 0.2695252954 | `51a03e33…0cea` |

x2 is the protected baseline: its checkpoint (Model A `6ff4a430…5b0b`), cached predictions (val_full `6c3a929f…2c03`,
test `fc4fdc8f…9191`, in `MOVD-opt07/outputs/opt07/x2_persistence_boost/`) and CSV are read only, hash-checked by
`scripts/opt08_loop.py verify`, and never written.

## Time contract (Europe/Kyiv, read from the system clock at every iteration)

* Before 07:20 on 2026-11-11: next defensible experiment, unless the repository is unsafe.
* From 07:20: no new experiment, training, inference, cache generation or long test. 07:20-07:55 closeout only.
  By 08:00 no OPT-08 Python / CUDA / background process runs.
* A job starts only if its conservative runtime leaves >= 30 min before 07:20 for evaluation and closeout. No new GPU
  job after 06:15 unless its whole runtime is < 30 min. A job at risk of crossing the cutoff is stopped at a
  checkpoint, logs kept. No iteration ends with an OPT-08 process still running (polls of <= 60 s).

## Robust evaluation (fixed before any candidate)

**Split** (`movd.opt08.split_clips`, `results/opt-08-overnight/split.json`): the 7 val_full clips sorted by
sha256(`"OPT-08 split v1:"` + clip name); the first 4 are **discovery**, the other 3 **confirmation**.
Discovery = 339, 086, 137, 182 (1,335 frames); confirmation = 305, 117, 268 (the 4K clip; 1,511 frames).

**Stages** (every experiment, also single-candidate ones):
* S1 — every frozen candidate of the experiment on the discovery clips only. Selection = highest discovery mAP
  (ties: earlier candidate). S1 passes if the selection's discovery mAP >= x2 + 0.0003.
* S2 — the selection, once, on the confirmation clips. Passes if mAP >= x2 (delta >= 0).
* S3 — val_full and its gate subsets, once, only after S1 and S2 pass.

**Upload eligibility** (deltas vs x2 on val_full, project evaluator, unrounded):

| ID | Criterion |
|---|---|
| U1 | val_full mAP >= x2 + 0.0003 |
| U2 | width <= 1920 subset (clips 086, 182, 305, 339; test-like) mAP >= x2 (+0.0005 preferred, used for ranking) |
| U3 | AP_small loss <= 0.003 |
| U4 | density >= 100 bin mAP loss <= 0.003 |
| U5 | at least 5 of 7 clips with per-clip delta >= 0 |
| U6 | gain does not depend on the 4K clip: val_full without `uav0000268_05773_v` mAP >= x2 + 0.0003 |
| U7 | deterministic: a separate process reproduces the val_full predictions byte for byte and the test predictions sha256 |
| U8 | evaluator, protocol configs, submission writer and validator unchanged vs `origin/develop` |

An eligible candidate gets one lossless test CSV, the project validator and the independent validator, and a record
of path, sha256, checkpoint, rule, rows and detections. **Nothing is uploaded.** At most 4 upload-ready CSVs.

## Ordered experiment directions (one per iteration)

| # | ID | What | Grid |
|---|---|---|---|
| 1 | `e1_x4_then_x2` | OPT-07 x4 then x2, both exactly as frozen in OPT-07 | 1 |
| 2 | `e2_tracklet_smoothing` | short-tracklet score smoothing (scores only), mutual-best x2 matching | 6 |
| 3 | `e3_broader_support` | one axis: which frames may support the x2 boost (t-+2, clip ends) | 4 |
| 4 | `e4_checkpoint_soup` | weight interpolation of Model A (epoch0.pt) with the adjacent epoch1.pt (`5594fb0e…5799`), then the CH-13 pipeline and x2 | 3 |
| 5 | `e5_lowlr_epoch` | optional: one extra epoch from Model A with one anti-overfitting change | 1 |
| 6 | `e6_combination` | only components that each passed S1-S3 and U1-U8 | 1 |

E1-E3 grids: `movd/opt08.py` EXPERIMENTS and `results/opt-08-overnight/<id>/rule.md`, committed together with this file.

**E4 (fixed now):** soup = w x Model A + (1 - w) x epoch1.pt for w in {0.8, 0.65, 0.5}, applied to every floating
tensor of the inference weights; rejected without any inference unless both files hash-match `checkpoints.json`, have
the same architecture, the same parameter names and the same shapes. Inference: the unchanged CH-13 pipeline
(6 views, NMS 0.5, top 500), then x2 (IoU 0.5) — the comparison with x2 keeps the post-processing equal. Cheap
CUDA smoke (a few frames) first; S1 on the discovery clips needs ~11 GPU min per weight. Exact rule in
`e4_checkpoint_soup/rule.md` before the first soup inference.

**E5 (fixed now, optional):** the CH-14 curve shows val_fast falling from 0.279 (epoch 1) to 0.2585 (epoch 2) while
the warmup still raised the learning rate (2.8e-4 to 4.6e-4). Hypothesis: the drop is driven by the rising LR, so one
epoch from Model A at a low constant LR may consolidate without that loss. Run only if it can finish (smoke,
1 epoch ~26 min, S1-S3 inference) with >= 30 min to spare before 07:20; exact config, runtime and stop rule in
`e5_lowlr_epoch/training_decision.md` before the CUDA smoke. No RT-DETR-x, no RT-DETR-l at 1600, no multi-epoch search.

**E6 (fixed now):** run only if at least two components (among the E1 x4 step, the E2/E3 selection as a replacement
of x2, the E4/E5 weights) each passed all stages and U1-U8. Order: weights, then x4 on the raw views (if E1 passed),
then the best passed temporal rule by val_full delta. Not run if fewer than two passed.

## Limits

One GPU process at a time, `nvidia-smi` before each GPU job, <= 4 GPU hours in total. Never: Kaggle upload or final
selection change, CH-14 / CH-16 / report.md / homework.ipynb edits, evaluator / ground truth / ignore policy / class
map / CSV format changes, committing data, checkpoints, raw predictions or CSVs, force-push or destructive git,
Co-Authored-By trailers. Negative results are kept.
