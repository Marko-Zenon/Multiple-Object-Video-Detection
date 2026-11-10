# OPT-03 — frozen experiment (committed before any candidate result)

Written after the val_fast diagnosis (`diagnosis.md`) and **before** any candidate below was evaluated.
It is not changed after the results are seen.

## Hypothesis

Boxes from a tile view that touch an inner edge of that tile (an edge that is not a frame edge) are mostly cut-off
parts of objects (val_fast: 47,158 scored boxes, precision@0.5 0.35 %). Removing or down-weighting them before the
CH-13 merge raises precision, and therefore mAP, without losing small-object recall, because the same objects are
seen whole by the full / flip views or by the overlapping neighbour tile (tiles overlap by 0.2 of the frame).

## Exact change

In `movd.infer.combine` order: decode every view (top-300, conf 0.001) → **new step: for each tile view, find boxes
whose x0 ≤ tile_x0 + m·tile_w (if tile_x0 > 0), y0 ≤ tile_y0 + m·tile_h (if tile_y0 > 0), x1 ≥ tile_x1 − m·tile_w
(if tile_x1 < W) or y1 ≥ tile_y1 − m·tile_h (if tile_y1 < H), and multiply their score by f (f = 0 removes them)** →
class-wise NMS 0.5 → global top 500. Full and flip views are never changed. Implemented in
`movd/opt03.py::edge_filter_combine`, test `tests/test_opt03.py`.

## Held constant

Checkpoint (Model A, sha256 `6ff4a430…5b0b`), the 6 views, imgsz 1344, tile grid 2×2 / frac 0.6, decoder top-300 and
conf 0.001, NMS method and IoU 0.5, the 500 cap, the evaluator, the frozen splits and the ignore policy. No training.
Same raw cache (`outputs/opt03/raw/`), so a candidate differs from Model A only by the new step.

## Candidate grid (6 of the allowed 12; one conceptual variable: how border boxes are treated)

| id | f (score factor) | m (margin, fraction of the tile side) |
|---|---:|---:|
| e1 | 0 (drop) | 0.002 |
| e2 | 0 (drop) | 0.005 |
| e3 | 0 (drop) | 0.01 |
| e4 | 0 (drop) | 0.02 |
| e5 | 0.5 | 0.005 |
| e6 | 0.25 | 0.005 |

## Gate to val_full (all on val_fast, against Model A's unrounded values)

* G-a mAP ≥ 0.2892166 + 0.003 = **0.2922166**
* G-b AP_small ≥ 0.1427339 − 0.003 = **0.1397339**
* G-c density bin ≥ 100 mAP ≥ 0.3515018 − 0.005 = **0.3465018**
* G-d projected test inference < 90 min (the new step is CPU post-processing on the same 6 passes; measured)

## Selection rule

Among the candidates passing G-a…G-d, select the highest val_fast mAP. If two are within 0.0005 mAP, prefer
drop (f = 0) over down-weighting, then the smaller margin. If none passes: negative result, stop; no val_full
candidate run, no CSV.

## Final gate (val_full, evaluated exactly once for the selected candidate)

* F-a mAP ≥ 0.2859619 + 0.003 = **0.2889619**
* F-b no per-class AP drop > 0.010 and no density-bin mAP drop > 0.010 vs Model A on val_full
  (`results/ch-14b-rtdetr-short-finetune/per_class_stageA_epoch1.csv`, `density_stageA_epoch1.csv`)
* F-c checkpoint, raw cache, config and prediction hashes recorded

If F-a…F-c pass: test predictions from the same raw-cache path (identity-checked against the live pipeline on a
subset), lossless CSV, project + independent validators. Never uploaded by Claude. If they fail: negative result.

## Stop conditions and budget

* Budget: CPU only for the 6 candidates; GPU only for the raw caches (val_fast, val_full, test ≈ 60 min, already
  running) and one same-session speed check. No training run (0 of the allowed 2).
* Stop if the cache identity check fails, if any candidate crashes in a way that changes the merge for non-tile
  views, or at 21:00 Kyiv (freeze target 21:30); then record what exists as PARTIAL.
* No Kaggle-based tuning; Kaggle is never queried.
