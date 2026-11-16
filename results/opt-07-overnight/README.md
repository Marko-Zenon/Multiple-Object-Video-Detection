# OPT-07 — overnight post-processing loop on protected Model A (2 promoted, 2 rejected, 1 not run)

Optional post-plan addendum requested by Badabuba on 2026-11-11. Worker: Claude (autonomous loop). Not report prose:
interpretation and conclusions are left to the students (TODOs at the end). No training, no new model, no ensemble
with Model F, no evaluator or protocol change, **no Kaggle upload**. Model A (checkpoint `6ff4a430…5b0b`, CSV
`e587c68b…0da7`, val_full 0.2859619, public 0.2774301362) is unchanged (`identity.json`).

Running log, gate and per-iteration details: [`progress.md`](progress.md). Machine-readable summary: [`summary.csv`](summary.csv).

## Why the gate is stricter than earlier cycles

OPT-06A c4 raised val_full to 0.2886 (+0.0026) but scored 0.2695252954 public (user-reported), below Model A. So a
val_full gain alone was not accepted here. Every candidate had to pass P1–P7 (fixed in `progress.md` before the first
result): val_full >= A + 0.0005; width <= 1920 subset (the test-like resolution) >= A + 0.0005; AP_small loss <= 0.003;
>=100 bin loss <= 0.005; >= 4 of 7 clips non-negative; without-4K-clip >= A + 0.0005; deterministic. Each candidate
modifies only Model A's own predictions, has one frozen parameter set (no grid, no val_fast screening) and was
evaluated exactly once.

## Results (val_full, deltas vs Model A, project evaluator)

| # | Candidate | Rule commit | mAP | Δ mAP | Δ w<=1920 | Δ w/o 4K | Δ AP_S | Δ >=100 | clips >= 0 | Decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | x1 temporal isolation demotion | 771b851 | 0.2862 | +0.0002 | +0.0015 | +0.0006 | -0.0016 | -0.0022 | 4/7 | rejected (P1) |
| 2 | **x2 temporal persistence boost** | 771b851 | **0.2879** | **+0.0019** | **+0.0026** | +0.0019 | +0.0019 | +0.0016 | **7/7** | **promoted** |
| 3 | x3 two-sided one-frame interpolation | 771b851 | 0.2860 | +0.00005 | +0.00001 | +0.00004 | +0.00004 | +0.0001 | 5/7 | rejected (P1, P2, P6) |
| 4 | **x4 OPT-03 e2 border drop, val_full confirmation** | 6101fc7 | **0.2874** | **+0.0014** | +0.0011 | +0.0014 | +0.0004 | +0.0019 | **7/7** | **promoted** |
| 5 | x5 combination (x4 then x2) | — | — | — | — | — | — | — | — | not run: 2-CSV cap reached |

Unrounded values: each `xN/gate.json`, `xN/subsets.csv` (all subsets and clips), `xN/per_class.csv`. Both promoted
candidates improve all 8 classes.

## Upload-ready CSVs (validated, NOT uploaded)

| | x2 persistence boost | x4 border drop (e2) |
|---|---|---|
| Path | `C:/UCU/AI/hw2/MOVD-opt07/submissions/opt07_x2_persistence_boost_test.csv` | `C:/UCU/AI/hw2/MOVD-opt07/submissions/opt07_x4_border_confirmation_test.csv` |
| sha256 | `9dc8fb03dc708b2b78b05afeff29289317a810f0219e9be4e2fd50029433c867` | `09a5c8d14633f4bade9b9c5b2edd6122c38ca8c3de0cf8afa6b5f79184cd2fcb` |
| Size / rows / detections | 172,652,039 B / 3,811 / 1,875,374 | 167,530,212 B / 3,811 / 1,820,997 |
| Test predictions sha256 | `fc4fdc8f…9191` (identical across two processes) | `bb7adfaa…6b8c` (identical across two processes) |
| Source | Model A cached test predictions (`5f9c4913…930e`), scores only | Model A test raw cache, identity-checked vs `5f9c4913…930e` (max diff 0) |
| Exact rule | [`x2_persistence_boost/rule.md`](x2_persistence_boost/rule.md) | [`x4_border_confirmation/rule.md`](x4_border_confirmation/rule.md) |
| Validators | project OK; independent valid, 0 rows differ | project OK; independent valid, 0 rows differ |

Suggested Kaggle descriptions:
`OPT-07 x2: Model A + temporal persistence boost (both-neighbour IoU>=0.5, score=max(s, 3-frame mean)); val_full 0.2879 (A 0.2860)` and
`OPT-07 x4: Model A + OPT-03 e2 inner tile-border drop (margin 0.005); val_full 0.2874 (A 0.2860)`.

## Recommended upload order (decision stays with the students)

1. **x2** first: the largest gain on val_full (+0.0019) and on the test-like width <= 1920 subset (+0.0026), and
   AP_small also rises (+0.0019); it changes only scores, so the box set equals the Model A submission's.
2. **x4** second: smaller val_full (+0.0014) and test-like (+0.0011) gains, but positive on every clip and the
   largest dense-scene gain (+0.0019).

Keep Model A (0.2774301362) as a final-selection fallback until a public score confirms either CSV. After an upload,
log the score in `results/submissions.csv`. The gains are small (+0.001–0.002) and the val/test domains differ
(OPT-06A), so a public score below Model A is possible.

## Runtime and hardware

Nazar's laptop. CPU for all candidates (transform 7–14 s per split; val_full evaluation of all 10 subsets ~4 min,
3 worker processes). RTX 5070 only for Model A raw shards: 40.4 of the allowed 90 min (val_full 3 clips 11.7 min,
test 28.7 min, 0.49 s per frame for 6 views), one process at a time, released.

## Deviations and caveats

* x4 is OPT-03 e2 selected on val_fast among 6 OPT-03 candidates, so it carries OPT-03's selection on val_fast;
  val_full was never used for it before. x1–x3 were each one predeclared setting, chosen without any metric.
* The order of the cache fill: loop iteration 4 ended while the val_full shard fill was running; iteration 5 finished
  x4. Each x4 step (cache, identity, gate, determinism, CSV) happened after its rule commit 6101fc7.
* `scripts/opt03_cache.py` writes its run record to `results/opt-03-modelA-diagnosis/cache/`; the two new records
  were moved to `x4_border_confirmation/cache_{val_full,test}.json` so OPT-03's evidence is unchanged.
* The OPT-06A public score 0.2695252954 was added to `results/submissions.csv` as user-reported.
* x5 is untested; nothing is claimed about whether x2 and x4 add up.

## Student TODOs (Claude does not write conclusions)

* TODO (students): upload decision for the two CSVs and the 2 final Kaggle selections; log scores in `results/submissions.csv`.
* TODO (students): interpret why the score-only persistence boost helps while isolation demotion and interpolation do not.
* TODO (students): decide whether OPT-07 is mentioned in the report.
