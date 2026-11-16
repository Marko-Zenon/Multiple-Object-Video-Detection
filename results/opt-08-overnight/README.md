# OPT-08 — overnight optimization loop on the OPT-07 x2 baseline (4 upload-ready, 2 rejected)

Optional post-plan addendum requested by Badabuba (Nazar) on 2026-11-11 as an autonomous loop. Worker: Claude.
This is evidence, not report prose: interpretation and conclusions are left to the students (TODOs at the end).
**No Kaggle upload, no change to the Kaggle final selections**, no evaluator / protocol / ground truth / ignore
policy / class map / CSV-format change. The protected x2 baseline (Model A `6ff4a430…5b0b` + OPT-07 persistence
boost; CSV `9dc8fb03…c867`, public 0.2805168313) is unchanged (`identity.json`).

Contract and frozen rules: [`TASK.md`](TASK.md). Running log: [`progress.md`](progress.md). Machine-readable:
[`summary.csv`](summary.csv). Robustness: [`robustness/`](robustness/).

## Protocol (fixed before the first candidate, 0e73d20)

* Clip split (`split.json`): the 7 val clips sorted by sha256(`"OPT-08 split v1:"` + name); **discovery** = 339,
  086, 137, 182 (1,335 frames), **confirmation** = 305, 117, 268 (4K) (1,511 frames).
* S1 every frozen candidate on discovery only, selection = best discovery mAP, pass if >= x2 + 0.0003; S2 the
  selection once on confirmation, pass if >= x2; S3 val_full once after S1 and S2.
* Upload gate vs x2: U1 val_full >= +0.0003; U2 width <= 1920 subset >= 0 (+0.0005 preferred); U3 AP_small loss
  <= 0.003; U4 >= 100 bin loss <= 0.003; U5 >= 5/7 clips non-negative; U6 without the 4K clip >= +0.0003; U7
  deterministic in a separate process; U8 evaluator / protocol / CSV unchanged. At most 4 upload-ready CSVs.

## Results (project evaluator; deltas vs x2; x2 val_full = 0.287867)

| # | Experiment (rule commit) | Grid | Selected | S1 disc. | S2 conf. | val_full | Δ val_full | Δ w<=1920 | Δ AP_S | Δ >=100 | clips >= 0 | Δ w/o 4K | Decision |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | E1 x4 then x2 (0e73d20) | 1 | c1 | +0.0015 | +0.0011 | 0.2892 | +0.0014 | +0.0011 | +0.0003 | +0.0018 | 7/7 | +0.0013 | **upload-ready** |
| 2 | E2 tracklet score smoothing (0e73d20) | 6 | t4 replace-mean k2 | +0.0029 | **-0.0062** | — | — | — | — | — | — | — | rejected (S2) |
| 3 | E3 broader support (0e73d20) | 4 | b3 five-frame mean | +0.0004 | +0.0005 | 0.2885 | +0.0006 | +0.0005 | +0.0003 | +0.0003 | 7/7 | +0.0006 | **upload-ready** |
| 4 | E4 checkpoint soup (35464d5) | 3 | s1 w = 0.8 | +0.0030 | +0.0236 | 0.2945 | +0.0067 | +0.0049 | +0.0149 | -0.0022 | 6/7 | +0.0069 | **upload-ready** |
| 5 | E5 one low-LR epoch (5714d84, 1b138bb) | 1 | l1 | **-0.0186** | — | — | — | — | — | — | — | — | rejected (S1) |
| 6 | E6 combination (5f495e2) | 1 | k1 | +0.0050 | +0.0255 | **0.2966** | **+0.0087** | **+0.0066** | **+0.0157** | -0.0003 | **7/7** | +0.0089 | **upload-ready** |

Discovery deltas of every grid member: `<exp>/screen.csv`; all subsets and clips: `<exp>/subsets.csv`; per class:
`<exp>/per_class.csv`; size and density bins: `<exp>/size_density.csv`; full metrics: `<exp>/val_full/metrics.json`.

**Per class (Δ AP vs x2).** E1 and E3 raise all 8 classes (E1 +0.0002 to +0.0027; E3 +0.0001 to +0.0014).
E4: truck +0.0175, tricycle +0.0228, bus +0.0137, car +0.0036, person +0.0029; bicycle -0.0020, van -0.0016,
motor -0.0036. E6: tricycle +0.0274, truck +0.0187, bus +0.0151, car +0.0061, person +0.0041, van +0.0008,
bicycle +0.0003; motor -0.0026.

**Size and density (Δ vs x2).** E4: AP_S +0.0149, AP_M +0.0056, AP_L +0.0035; density 0-19 +0.0107, 20-49
+0.0094, 50-99 +0.0023, >=100 -0.0022. E6: AP_S +0.0157, AP_M +0.0072, AP_L +0.0071; density 0-19 +0.0131,
20-49 +0.0107, 50-99 +0.0047, >=100 -0.0003. E1 and E3 are positive in every size and density bin.

**Robustness (closeout, `robustness/`).** Leave-one-clip-out val_full (each of the 7 clips removed in
turn, x2 on the same frames; `robustness/leave_one_clip_out.csv`): every candidate stays above x2 for all 7
removals: E6 +0.0058 to +0.0094, E4 +0.0035 to +0.0072, E1 +0.0013 to +0.0016, E3 +0.0005 to +0.0007. All 4
CSVs and their test predictions re-hash to the recorded sha256 (`robustness/summary.json`). Evaluation only;
nothing was selected from these numbers.

## Rejected directions and why

* **E2 short-tracklet score smoothing** (6 frozen candidates): the discovery winner t4 (replace scores with the
  ±2-frame tracklet mean, the only variant that can lower scores) gained +0.0029 on discovery but lost -0.0062 on
  confirmation, so it was stopped at S2 without a val_full run. The boost-only variants were +0.0004 to +0.0015
  on discovery.
* **E5 one extra epoch from Model A at AdamW constant lr0 5e-5** (training_decision.md): finite loss, 26.5 min,
  weights updated, but discovery -0.0186 vs x2, rejected at S1. A lower LR alone did not avoid the post-epoch-1 drop.
* E3's other variants (gap-tolerant, clip-end one-sided, both) were within ±0.00004 on discovery (not selected).
* E4's w065 (+0.0024) and w050 (+0.0001) were not selected (S1 picks one).

## Upload-ready CSVs (validated by the project and the independent validator, **NOT uploaded**)

| Rank | CSV (absolute path) | sha256 | Weights | Exact rule | Rows / detections | val_full (Δ vs x2) |
|---:|---|---|---|---|---|---|
| 1 | `C:/UCU/AI/hw2/MOVD-opt08/submissions/opt08_e6_combination_k1_test.csv` | `35a225c905ac56ed86532314046821d10085f73781666ee843fb4c6d28127c35` | soup w080 `2d60d3fa…d939` | soup -> CH-13 merge with OPT-03 e2 border drop -> E3 b3 ([rule](e6_combination/rule.md)) | 3,811 / 1,841,598 | 0.2966 (+0.0087) |
| 2 | `C:/UCU/AI/hw2/MOVD-opt08/submissions/opt08_e4_checkpoint_soup_s1_test.csv` | `7dc9b318e354d4839b5dffae8f5973dd7da234fdbeb3c5f84057694878852014` | soup w080 `2d60d3fa…d939` | soup -> unchanged CH-13 merge -> x2 ([rule](e4_checkpoint_soup/rule.md)) | 3,811 / 1,881,202 | 0.2945 (+0.0067) |
| 3 | `C:/UCU/AI/hw2/MOVD-opt08/submissions/opt08_e1_x4_then_x2_c1_test.csv` | `17273d344bb0fed82804b69abba3d018ef89e8c04cc87291e78356f2b8f2dbf8` | Model A `6ff4a430…5b0b` | OPT-07 x4 -> x2 ([rule](e1_x4_then_x2/rule.md)) | 3,811 / 1,820,997 | 0.2892 (+0.0014) |
| 4 | `C:/UCU/AI/hw2/MOVD-opt08/submissions/opt08_e3_broader_support_b3_test.csv` | `e45f1b5416cae645100ae3d744d93511d6608e8b2b8c65d119ca3ff942c1a444` | Model A `6ff4a430…5b0b` | Model A -> five-frame-mean boost ([rule](e3_broader_support/rule.md)) | 3,811 / 1,875,374 | 0.2885 (+0.0006) |

Each `<exp>/test/run.json` records the CSV and test-prediction sha256, the checkpoint identity (the soup's own hash
for E4/E6), the writer counts and both validators' exit codes; `<exp>/determinism.json` the separate-process rerun;
`e4_checkpoint_soup/gpu_recheck_w080.json` the GPU re-inference check. The soup checkpoint itself is
`C:/UCU/AI/hw2/MOVD-opt08/outputs/opt08/soup/soup_w080.pt` (git-ignored; rebuildable with `scripts/opt08_soup.py build`).

Suggested Kaggle descriptions: `OPT-08 E6: soup 0.8 A + 0.2 epoch2, border drop, 5-frame boost; val_full 0.2966 (x2 0.2879)`,
`OPT-08 E4: soup 0.8 A + 0.2 epoch2 + x2; val_full 0.2945`, `OPT-08 E1: x4 border drop + x2; val_full 0.2892`,
`OPT-08 E3: 5-frame persistence boost; val_full 0.2885`.

## Recommended morning upload order (the decision stays with the students)

1. **E6** — largest val_full (+0.0087) and test-like (+0.0066) gains, 7/7 clips, dense bin flat.
2. **E4** — isolates the soup (+0.0067); if E6 scores below E4 publicly, the border / five-frame steps do not transfer.
3. **E1** — post-processing only on Model A (+0.0014), the safest change: same model as the public 0.2805 x2.
4. **E3** — smallest gain (+0.0006); upload only if a slot is left.

**Keep the OPT-07 x2 CSV (`9dc8fb03…c867`, public 0.2805168313) as a protected final selection** until a public
score beats it; Model A (0.2774301362) remains the conservative fallback. OPT-06A (val_full +0.0026, public below
Model A) shows that a val_full gain can fail to transfer: confirm E6 / E4 publicly before replacing x2. After each
upload, log the score in `results/submissions.csv`.

## Limitations and possible validation overfitting

* Only 7 val clips (2,846 frames): the soup gain is uneven by clip (E4: 117 +0.021, 305 +0.013, 268 +0.011 vs 086
  +0.0008, 137 -0.0016). The discovery / confirmation split protects selection, not the size of the gain.
* The soup weight (0.8) and the E3 mode were selected on discovery; confirmation and val_full were each used once
  per experiment, but val_full also includes the discovery clips, so the val_full deltas are not fully held-out.
* E6 is a combination of components selected on the same val data; its gate is not independent of E1, E3, E4.
* The test set differs from val (resolution mix, scenes); OPT-06A's val_full gain did not transfer publicly.
* E4/E6 use a new checkpoint (the soup) that was never trained directly; motor (and for E4 bicycle / van) AP falls.
* Ultralytics' trainer warns that `grid_sampler_2d_backward` is nondeterministic, so E5 training is not bit-exact
  reproducible (inference determinism was checked: GPU re-inference identical).

## Deviations

* E5: after the smoke, AdamW beta1 was pinned to Model A's 0.9 (explicit optimizer default 0.937), and a no-update
  stop rule was added (1b138bb), before the full run and before any E5 result.
* U8: the first E6 gate counted OPT-08's two added E5 run configs as a protocol change; the check was narrowed to
  exclude OPT-08's own added run configs and still fails on any modified frozen file (ce90662); only that flag was
  recomputed (`e6_combination/gate_u8_initial.json` keeps the original record).
* `scripts/train_ultra.py` wrote `results/opt08_e5_lowlr{,_smoke}/` (run config, hardware, runtime, Ultralytics logs).

## Runtime and hardware

Nazar's laptop, RTX 5070 Laptop 8 GB, one GPU process at a time, `nvidia-smi` checked before each job. GPU 111.7 of
240 min: E4 smoke 0.2, soup discovery caches 30.3, w080 confirmation 12.2, GPU recheck 1.4, w080 test 28.6; E5
smoke 2.4, epoch 26.5, discovery 10.1. Everything else on the CPU (val_full gate of all 12 subsets ~4-5 min).

## Manual actions remaining for Nazar (students only)

* TODO: decide which of the 4 CSVs to upload (suggested order above) within the daily limit; record each public
  score in `results/submissions.csv` (Claude did not upload).
* TODO: choose the 2 Kaggle final selections in the web UI (keep x2 protected unless a new CSV beats 0.2805168313).
* TODO: interpret why the soup helps small / large vehicles but not motor, and why the E2 / E5 discovery results
  did not transfer; decide whether OPT-08 is mentioned in the report.
