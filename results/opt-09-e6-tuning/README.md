# OPT-09 — focused local search around E6 (NEGATIVE result: no candidate passes the gate, no new CSV)

This is an optional post-plan addendum. Badabuba (Nazar) requested it on 2026-11-11 and Claude did the work.
This file is evidence, not report prose. Interpretation is left to the students (see the TODOs at the end).
Nothing was uploaded and the Kaggle final selections were not changed. The evaluator, protocol, ground truth,
ignore policy, class map and CSV format were not changed either, and there was no training.
The protected E6 artifacts were only read (`identity.json`).

Rules (frozen at 431a13d before any evaluation): [`selection_rule.md`](selection_rule.md). Log:
[`progress.md`](progress.md). Machine-readable: [`summary.csv`](summary.csv).

## Baseline: E6 (unchanged)

E6 = soup w080 (0.8 × Model A `epoch0.pt` + 0.2 × `epoch1.pt`, `2d60d3fa…d939`) → CH-13 merge with inner
tile-border boxes dropped (margin 0.005) → five-frame persistence boost (IoU 0.5, strength 1).
val_full 0.296583, discovery 0.334488, confirmation 0.270895. CSV `35a225c9…7c35`, Kaggle public
**0.2947349001** (user-reported). OPT-09 rebuilt E6 val_full byte-identically (`bdb2806e…ad52`) and re-evaluated it
on every subset; the results equal the OPT-08 record.

## Results (project evaluator; deltas are vs E6 on the same frames)

Split: the frozen OPT-08 split. Discovery = 339, 086, 137, 182; confirmation = 305, 117, 268 (4K).

| Phase | Candidate (one change) | Δ discovery | Δ confirmation | Δ val_full | Decision |
|---|---|---:|---:|---:|---|
| A1 border | m1 margin 0.0035 | -0.000053 | — | — | not selected |
| A1 border | m2 margin 0.0065 | +0.000043 | — | — | not selected |
| A1 border | m3 margin 0.008 | +0.000045 | — | — | **rejected S1** (< +0.0003) |
| A2 persistence | p1 strength 0.75 | -0.000420 | — | — | not selected |
| A2 persistence | p2 strength 1.25 | +0.000207 | — | — | **rejected S1** |
| A2 persistence | p3 strength 1.5 | -0.001177 | — | — | not selected |
| A2 persistence | p4 IoU 0.45 | -0.000319 | — | — | not selected |
| A2 persistence | p5 IoU 0.55 | +0.000008 | — | — | not selected |
| B soup | w1 w = 0.75 | **+0.000600** | **+0.003740** | — | component passed (S1, S2) |
| B soup | w2 w = 0.85 | -0.000104 | — | — | not selected |
| B soup | w3 w = 0.90 | -0.001009 | — | — | not selected |
| C final | c1 = E6 with soup w075 | +0.000600 | +0.003740 | **+0.000817** | **rejected (gate G1, G4)** |

c2 (a combination) was not formed because only one component passed.

### c1 promotion gate (`c_final/c1/gate.json`)

| Check | Rule | c1 | Pass |
|---|---|---:|---|
| G1 | val_full ≥ E6 + 0.0010 | 0.297399 (+0.000817) | **no** |
| G2 | width ≤ 1920 ≥ E6 + 0.0005 | 0.319111 (+0.000985) | yes |
| G3 | ≥ 5/7 clips non-negative | 6/7 (137: -0.0001) | yes |
| G4 | without 4K clip ≥ E6 + 0.0010 | 0.322095 (+0.000683) | **no** |
| G5 | AP_small loss ≤ 0.002 | 0.169247 (+0.001332) | yes |
| G6 | ≥100-object bin loss ≤ 0.002 | 0.350786 (-0.000255) | yes |
| G7 | 8 class results | `per_class.csv` | yes |
| G8 | determinism | not run (gate failed first) | — |
| G9 | evaluator / protocol / CSV unchanged | `git diff origin/develop` empty | yes |

Per clip (Δ vs E6): 086 +0.0019, 117 +0.0030, 137 -0.0001, 182 +0.0008, 268 +0.0008, 305 +0.0038, 339 +0.0024.
Per class (Δ AP): tricycle +0.0043, truck +0.0028, bicycle +0.0006, car +0.0006, bus +0.0001, van +0.0001,
person -0.0003, motor -0.0016. AP50 +0.0016, AP75 +0.0012, AP_M +0.0010, AP_L -0.0008; density 0-19 +0.0008,
20-49 +0.0014, 50-99 +0.0003, ≥100 -0.0003 (`size_density.csv`).

## Conclusion (factual)

* The border margin and the persistence boost are flat or worse around E6's values, so E6's settings are kept.
* Soup weight 0.75 is the only positive direction. It improves every subset except clip 137, but the val_full gain
  (+0.0008) is below the +0.0010 needed to justify another Kaggle submission. By the frozen rule, this is a
  negative result: no test inference, no CSV, and the gate was not weakened.
* Recommendation: keep **E6** (public 0.2947349001) and **E4** (0.2913788894) as the two Kaggle final selections.

## Runtime and hardware

The work ran on Nazar's laptop (RTX 5070 Laptop 8 GB), from 10:30 to about 11:55 Kyiv, well inside the 3-hour budget.
The GPU ran one process at a time and `nvidia-smi` was checked before each job. GPU time was 43.7 minutes:
smoke 0.6, discovery caches 31.0 (w075, w085, w090), w075 confirmation 12.1. No GPU time went to test inference.
All other steps ran on the CPU.

## Deviations

* The claim row first said 08:30 Kyiv because Git Bash ignored `TZ=Europe/Kyiv` and printed UTC. The real claim
  time was 10:30 Kyiv. The board row is corrected in the closing commit.
* The Phase B discovery caches (GPU inference only) ran while the Phase A screens ran on the CPU, so the work
  fit the budget. No Phase B result was evaluated before Phase A was decided. The selection rule allowed this
  in advance.
* E6 has no explicit score multiplier. Its implicit boost strength is 1 (the score is raised all the way to the
  mean). The A2 strength sub-grid therefore scales that boost: λ in own + λ(mean − own), with λ = 1 equal to E6
  exactly (`tests/test_opt09.py`).

## Files

Tracked here: `identity.json`, `baseline/`, `a1_border_margin/`, `a2_persistence/`, `b_soup_weight/` (screen,
confirm, soup_build, smoke) and `c_final/` (plan, c1 gate, subsets, per class, size/density, val_full metrics).
Git-ignored: the soups and raw caches under `MOVD-opt09/outputs/opt09/`. They can be rebuilt with
`scripts/opt09_soup.py build` and `cache`.

## TODO (students)

* TODO: decide whether OPT-09 goes in the report. If it does, interpret why weight 0.75 helps tricycle and truck
  but not motor, and why the boost is at a local optimum.
* TODO: confirm the 2 Kaggle final selections (currently E6 and E4, user-reported).
