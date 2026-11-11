# OPT-08 — progress log

Read this file, [`TASK.md`](TASK.md) and `git log` at the start of every iteration. Times are Europe/Kyiv.

## Status

| # | Experiment | Rule commit | S1 discovery | S2 confirmation | S3 val_full / gate | CSV | Decision |
|---|---|---|---|---|---|---|---|
| 1 | e1_x4_then_x2 | 0e73d20 | c1 +0.0015 pass | +0.0011 pass | 0.2892 (+0.0014), U1-U8 pass | `17273d34…dbf8` | **UPLOAD-READY 1** |
| 2 | e2_tracklet_smoothing | 0e73d20 | t4 +0.0029 pass | -0.0062 **fail** | not run (rule) | — | **rejected (S2)** |
| 3 | e3_broader_support | 0e73d20 | b3 +0.0004 pass | +0.0005 pass | 0.2885 (+0.0006), U1-U8 pass | `e45f1b54…c444` | **UPLOAD-READY 2** |
| 4 | e4_checkpoint_soup | 35464d5 | s1 (w080) +0.0030 pass | +0.0236 pass | 0.2945 (+0.0067), U1-U8 pass | `7dc9b318…2014` | **UPLOAD-READY 3** |
| 5 | e5_lowlr_epoch | decision pending | — | — | — | — | pending (optional) |
| 6 | e6_combination | 5f495e2 | — | — | — | — | pending |

Upload-ready CSVs: 3 of 4. GPU minutes used: 72.7 of 240.

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
  1,820,997 detections (0 rejected, 0 over the cap); project validator OK, independent validator valid (0 rows differ). Not uploaded. (02:17)

### Iteration 2 — 2026-11-11 02:17-02:25

* State: branch clean, OPT-08 row still IN_PROGRESS (mine), notebook lock FREE, GPU idle, no Python process.
* **E2 `e2_tracklet_smoothing`** (rule 0e73d20, 6 candidates, CPU, 6.1 min screen). S1 discovery deltas vs x2:
  t1 +0.000696, t2 +0.001113, t3 +0.001520, t4 +0.002879, t5 +0.000417, t6 +0.000679. Selected t4 (`replace_mean`, k=2; the only rule that can lower scores), S1 pass.
  S2 confirmation (305, 117, 268): t4 -0.006218 vs x2 -> **rejected at S2**; no val_full run, no CSV.
  The discovery gain did not transfer to the other clips (selection overfit to the discovery clips; kept as a
  negative result). No retuning. E2 cannot enter E6.

### Iteration 3 — 2026-11-11 02:25-02:39

* State: clean, row mine, lock FREE, GPU idle, no Python process.
* **E3 `e3_broader_support`** (rule 0e73d20, 4 candidates, CPU). S1 discovery deltas vs x2: b1 +0.000024, b2 -0.000039, b3 +0.000369, b4 -0.000013.
  Selected b3 (`five_frame_mean`: x2 condition, mean also over matched t-2 / t+2), S1 pass (just above +0.0003).
  S2 confirmation +0.000504 (pass). S3 val_full 0.288472
  (+0.000605 vs x2); width<=1920 +0.000547 (preferred +0.0005 met); AP_small +0.000322;
  >=100 +0.000324; 7/7 clips >= 0 (smallest 305 +0.00004); without 4K +0.000580; U7 identical
  (val_full in a fresh process, test across two processes); U8 unchanged. **Upload-ready CSV 2 of 4**:
  `C:/UCU/AI/hw2/MOVD-opt08/submissions/opt08_e3_broader_support_b3_test.csv`, sha256 `e45f1b5416cae645100ae3d744d93511d6608e8b2b8c65d119ca3ff942c1a444`, 172,647,586 B, 3,811 rows,
  1,875,374 detections (box set = Model A / x2, scores only); both validators pass. Not uploaded.
  Small gain (about a third of E1's); the gap-tolerant and clip-end variants were neutral or negative on discovery.
* E6 now has two independently passed components (E1's x4 step and E3 b3 as the temporal rule), so by TASK.md it
  will be x4 -> b3; it runs after E4 / E5 (order), rule written before it runs.

### Iteration 4 — 2026-11-11 02:39- (E4 checkpoint soup)

* State: clean, row mine, lock FREE, GPU idle before each GPU job (checked with nvidia-smi), one GPU process at a time.
* Compatibility (`compatibility.json`, before any inference): both hashes = `checkpoints.json`, same class / YAML /
  941 tensor names, shapes, dtypes, adjacent epochs -> compatible. Rule 35464d5 committed before any soup was built.
* CUDA smoke (`smoke.json`): Model A through the new cache + merge path on 20 frames = Model A's stored predictions
  exactly (9,940 detections, max diff 0), 0.5 s/frame, 0.86 GB peak. Soups built (`soup_build.json`).
* S1 (discovery raw caches, 30.3 GPU min for 3 soups): s1 (w080) +0.003009, s2 (w065) +0.002447, s3 (w050) +0.000088. Selected s1 = w080, pass.
* S2 (w080 confirmation cache, 12.2 GPU min): +0.023630 vs x2 (pass; discovery predictions rebuilt
  from the caches equal S1's byte for byte).
* S3 val_full 0.294539 vs x2 0.287867 (**+0.006672**); width<=1920 +0.004948;
  AP_small **+0.014874**; >=100 bin -0.002151 (allowed -0.003); 6/7 clips >= 0 (137 -0.0016);
  without 4K +0.006905. Per class: person +0.0029; bicycle -0.0020; car +0.0036; van -0.0016; truck +0.0175; tricycle +0.0228; bus +0.0137; motor -0.0036.
  Caveat for the students: the gain is much larger than any post-processing gain and is uneven by clip
  (117 +0.021, 305 +0.013, 268 +0.011 vs 086 +0.0008, 137 -0.0016); it rests on 7 val clips.
* U7 GPU part (`gpu_recheck_w080.json`): re-inference of clip 305 in a new process = cached shard exactly (max diff 0).
* Test raw cache for w080 started (3,811 frames, ~31 GPU min).
* Test raw cache 28.6 GPU min. U7 CPU part: val_full rebuilt in a fresh process byte-identical
  (`ba88e072…9d4a`), test predictions identical across two processes (`f97dd8e1…43e9`). **Upload-ready CSV 3 of 4**:
  `C:/UCU/AI/hw2/MOVD-opt08/submissions/opt08_e4_checkpoint_soup_s1_test.csv`, sha256 `7dc9b318e354d4839b5dffae8f5973dd7da234fdbeb3c5f84057694878852014`, 173,452,716 B, 3,811 rows, 1,881,202 detections;
  checkpoint = soup w080 sha256 `2d60d3fa66d9fd37032bcaaaeb297e98510217b600b5809f73873b36360cd939` (recorded in run.json and checked by the independent
  validator); project validator OK, independent validator valid (0 rows differ). Not uploaded. (04:39)
  GPU total so far 72.7 min (smoke 0.2, discovery 30.3, confirmation 12.2, recheck 1.4, test 28.6).
