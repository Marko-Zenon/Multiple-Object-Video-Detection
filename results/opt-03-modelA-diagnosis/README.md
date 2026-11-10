# OPT-03 — Model A diagnosis + one predeclared intervention (NEGATIVE result)

Optional post-plan addendum requested by Badabuba on 2026-11-10. Worker: Claude. Not report prose: interpretation for
the report is left to the students (TODOs at the end). No training, no Kaggle upload, no new CSV.
Model A (CH-14b epoch-1 checkpoint + CH-13 pipeline, val_full 0.2860, public 0.2774301362) and its CSV
(sha256 `e587c68b…0da7`) stay selected and were not touched.

## Order of events (branch `opt-03-modelA-diagnosis`)

1. Raw per-view cache of Model A on val_fast (`scripts/opt03_cache.py`; RTX 5070, 0.47 s per frame for 6 views,
   peak 0.65 GiB allocated; `cache/val_fast.json`). Identity: re-merging the cache reproduces the stored CH-14b
   val_fast predictions exactly (352,886 boxes, max difference 0.0, mAP 0.2892166; `diagnosis/identity.json`).
2. Diagnosis on val_fast only (`diagnosis.md`, `diagnosis/`). Main finding: precision, not the 500 cap (no-cap oracle
   +0.00006); the clearest FP group is tile boxes on inner tile borders (13 % of scored boxes, precision 0.35 %).
3. `selection_rule.md` (hypothesis, exact change, constants, 6-candidate grid, gates, stop rules) committed in
   **528167a before any candidate was evaluated**.
4. Screen (`scripts/opt03_edge.py screen`): no candidate passes G-a. Per the rule: stop. val_full was **not** evaluated
   for any candidate, no test predictions, no CSV.

## val_fast screen (714 frames, project evaluator, maxDets 500)

Gate: G-a mAP ≥ 0.2922166, G-b AP_S ≥ 0.1397339, G-c ≥100-bin ≥ 0.3465018, G-d test < 90 min.

| id | border boxes | margin | mAP | Δ vs A | AP50 | AP_S | ≥100 bin | scored boxes | frames at 500 | passes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Model A | kept | — | 0.2892 | — | 0.5397 | 0.1427 | 0.3515 | 352,886 | 599 | — |
| e1 | dropped | 0.002 | 0.2902 | +0.0010 | 0.5409 | 0.1430 | 0.3521 | 348,763 | 548 | no (G-a) |
| e2 | dropped | 0.005 | 0.2906 | +0.0014 | 0.5414 | 0.1432 | 0.3530 | 347,614 | 542 | no (G-a) |
| e3 | dropped | 0.01 | 0.2906 | +0.0014 | 0.5414 | 0.1432 | 0.3531 | 346,933 | 540 | no (G-a) |
| e4 | dropped | 0.02 | 0.2905 | +0.0013 | 0.5412 | 0.1432 | 0.3530 | 345,836 | 533 | no (G-a) |
| e5 | score × 0.5 | 0.005 | 0.2906 | +0.0014 | 0.5417 | 0.1432 | 0.3530 | 352,913 | 600 | no (G-a) |
| e6 | score × 0.25 | 0.005 | 0.2906 | +0.0014 | 0.5417 | 0.1433 | 0.3530 | 352,928 | 600 | no (G-a) |

Every candidate is above Model A on every class (`screening_val_fast.csv`, per class and per density bin), on AP_S
and on the dense bin, so the change is safe but small: the best is less than half of the predeclared +0.003.
Projected test time is unchanged (29.0 min: same 6 passes, +2.5–3.2 ms CPU per frame). Selection:
`selection_val_fast.json` (`passing = []`, `selected = null`).

Fact relevant to the size of the gain (from the same files): dropping the border boxes lowers the number of
scored boxes by only 4–7 k of 352,886, because the freed slots under the 500 cap are refilled by the next boxes
below rank 500, which are also low-score FPs (`diagnosis/tables.json` → `below_cap`: 78 TPs among 195,544 cut boxes).

## Reproduce (CPU, from the ignored raw cache)

    python scripts/opt03_cache.py --split val_fast --weights <MOVD-ch14>/outputs/final_rtdetr/train/weights/epoch0.pt --device 0
    python scripts/opt03_diagnose.py identity
    python scripts/opt03_diagnose.py ablations
    python scripts/opt03_diagnose.py tables
    python scripts/opt03_edge.py screen

Determinism: `tables` and `screen` were run twice; `tables.json` and all six `metrics.json` are byte-identical;
`screening_val_fast.csv` differs only because it also stores CPU timing (`determinism.json`).
`opt03_edge.py confirm / test / live` exist for the passing case and were not run (rule).

## Not completed / deviations

* The val_full and test raw caches were started in the background before the screen (to save time if a candidate
  passed) and stopped after the negative screen: val_full was partially cached (ignored `outputs/opt03/raw/val_full/`
  shards), never merged and never evaluated; the test cache was never started. No val_full number of OPT-03 exists.
* Selection bias: the hypothesis was chosen on Model A's val_fast tables, and screening was on the same val_fast.
  This could only inflate the screen; the screen still failed.
* The diagnostic matcher (`movd/opt03.py::match_frame`) is a simplified IoU-0.5 matcher for the breakdown tables;
  all mAP values are from `movd/evaluate.py`.

## Student TODOs (Claude does not write conclusions)

* TODO (students): explain why removing 13 % of the scored boxes, almost all FPs, moves mAP by only +0.0014
  (where in the PR curve do these boxes sit, and what refills the 500 slots?).
* TODO (students): decide whether to mention OPT-03 in the report as a negative inference experiment.
