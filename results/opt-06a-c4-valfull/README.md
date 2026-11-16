# OPT-06A — OPT-02 candidate c4 on val_full (cached predictions, CPU only)

Rule (committed before any result, 8b6c1df): [`selection_rule.md`](selection_rule.md).
Candidate: OPT-02 `c4_w075_nms06` unchanged: Model A x 1.0 + Model F (OPT-02 "B", CH-11 rtdetr_baseline + CH-13) x 0.75,
class-wise NMS 0.6, global top 500, merged by `movd/ensemble.py` through `scripts/opt02_ensemble.py`'s `merge`.
CLI: `scripts/opt06a_c4.py verify | reproduce | confirm | test`.

## Result

| Step | Outcome |
|---|---|
| Identity | all six cached inputs match `results/opt-02-rtdetr-ensemble/identity.json`; both checkpoints hashed (not loaded) and match ([`identity.json`](identity.json)) |
| val_fast reproduction | mAP 0.29209989893432436 = registered OPT-02 value; metrics, per-class, density and prediction count identical ([`reproduce_val_fast.json`](reproduce_val_fast.json)) |
| val_full (once) | c4 0.2886 vs Model A 0.2860 (+0.0026) ([`comparison_val_full.csv`](comparison_val_full.csv), [`gate_val_full.json`](gate_val_full.json)) |
| Competition gate | **passes** (K-a mAP 0.2886 >= 0.2840; K-b AP_S +0.0014 >= -0.005; K-c >=100 bin +0.0008 >= -0.010) |
| Test CSV | written from the cached A and F test predictions (no inference); project and independent validators pass; **not uploaded** |

### val_full, c4 vs Model A (unrounded values in `comparison_val_full.csv`)

| Metric | Model A | c4 | delta |
|---|---:|---:|---:|
| mAP | 0.2860 | 0.2886 | +0.0026 |
| AP50 | 0.5358 | 0.5403 | +0.0044 |
| AP75 | 0.2600 | 0.2625 | +0.0025 |
| AP_small | 0.1504 | 0.1517 | +0.0014 |
| AP_medium | 0.2568 | 0.2571 | +0.0002 |
| AP_large | 0.4812 | 0.4827 | +0.0015 |
| person | 0.3004 | 0.3014 | +0.0010 |
| bicycle | 0.2509 | 0.2537 | +0.0029 |
| car | 0.4536 | 0.4596 | +0.0060 |
| van | 0.3275 | 0.3272 | -0.0003 |
| truck | 0.3080 | 0.3091 | +0.0011 |
| tricycle | 0.2561 | 0.2657 | +0.0096 |
| bus | 0.2399 | 0.2400 | +0.00004 |
| motor | 0.1512 | 0.1521 | +0.0008 |
| density 0-19 | 0.2081 | 0.2153 | +0.0072 |
| density 20-49 | 0.2981 | 0.3021 | +0.0040 |
| density 50-99 | 0.3073 | 0.3110 | +0.0037 |
| density >=100 | 0.3497 | 0.3505 | +0.0008 |

16 of 17 secondary numbers (AP50, AP75, AP_S/M/L, 8 classes, 4 density bins) improve; only van drops (-0.0003). The gain (+0.0026) is below the stricter OPT-02 val_full
gate (+0.003), so c4 is **not** the internal selection over Model A; the CSV is a Kaggle exploration candidate only.

## Candidate CSV (not uploaded)

| | |
|---|---|
| Path | `C:/UCU/AI/hw2/MOVD-opt06a/submissions/opt06a_c4_ensemble_test.csv` (git-ignored) |
| Size | 175,459,286 bytes, 3,811 rows, 1,905,500 detections (every frame at 500) |
| sha256 | `51a03e33ca487fa2bd4c78948b0bdd71bd57ec59a35f456b8c303c1c25ba0cea` |
| Merged test predictions | `outputs/opt06a/test/test_predictions.json`, sha256 `b168ede4…7f4b6c` |
| Inputs | A test `5f9c4913…87930e`, F test `184ddccc…647ccd`; 18.2 % of the kept boxes come from F |
| Validators | `scripts/validate_submission.py` OK; `scripts/check_submission_independent.py` valid (0 rows differ from the predictions) ([`test/`](test/)) |

Suggested Kaggle description: `OPT-06A: RT-DETR-l ensemble, Model A (CH-14b ep1) x1.0 + F (CH-11 baseline) x0.75, CH-13 pipeline, class-wise NMS 0.6, top 500; val_full 0.2886 (A 0.2860)`

## Runtime and hardware

Nazar's laptop, CPU only; RTX 5070 not used; no checkpoint loaded, no inference rerun. verify about 5 s; val_fast
reproduction 59 s; val_full (A + c4 evaluation, merge 6.2 s) 6.5 min, peak process memory 5.4 GB; test merge (7.6 s) +
CSV + both validators 1.4 min.

## Caveats

- Inputs are the capped CH-13 pipeline outputs (as in OPT-02).
- c4 was chosen on val_fast among seven OPT-02 candidates, so its val_fast number is optimistic; val_full is the
  single held-out check. Validation and test domains differ, so the Kaggle score may not follow val_full.

TODO (students): interpretation and conclusions for the report.
