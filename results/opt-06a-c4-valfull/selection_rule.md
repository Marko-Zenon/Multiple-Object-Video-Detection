# OPT-06A — frozen rule: OPT-02 candidate c4 on val_full

Written and committed **before any OPT-06A metric was produced** (2026-11-10, Badabuba / worker Claude).
Hypothesis, candidate and gate are fixed by the student's request and are not changed after results are seen.

## Hypothesis

OPT-02 candidate `c4_w075_nms06` was rejected only because its val_fast gain was +0.0029 instead of the +0.003 gate
(0.2921 vs Model A 0.2892). Kaggle submissions are no longer scarce (student-reported 10/day, G12 still open) and the
val and test domains differ, so c4 gets one direct val_full check and, if it passes the gate below, one test CSV.

## Candidate (frozen, no retuning)

Exactly OPT-02 `c4_w075_nms06` from `results/opt-02-rtdetr-ensemble/candidates.json`, merged with the unchanged
OPT-02 code (`movd/ensemble.py`, `scripts/opt02_ensemble.py` `merge`):

| Model | Checkpoint sha256 | Score multiplier |
|---|---|---:|
| A (CH-14b epoch 1 + CH-13 pipeline) | `6ff4a430…5b0b` | 1.0 |
| F (CH-11 rtdetr_baseline + CH-13 pipeline; OPT-02 "Model B") | `da5d3625…e872` | 0.75 |

Union per frame, class-wise NMS IoU 0.6, global top 500 per frame, boxes passed through unchanged. Inputs are the same
cached capped pipeline predictions as OPT-02 (hashes checked against `results/opt-02-rtdetr-ensemble/identity.json`).
No inference is rerun, no model is loaded, no GPU is used.

## Steps and stop conditions

1. Identity: all six cached inputs (A/F x val_fast/val_full/test) must match the OPT-02 hashes. Otherwise stop.
2. Reproduce: c4 on val_fast must reproduce the registered OPT-02 metrics
   (`results/opt-02-rtdetr-ensemble/val_fast/c4_w075_nms06/metrics.json`, mAP 0.29209989893432436) exactly
   (metrics, per-class and density tables). Otherwise stop.
3. val_full: c4 is evaluated on val_full **once**, against Model A evaluated from its cached val_full predictions
   (registered mAP 0.28596193196293734).

## Competition-oriented CSV gate (val_full, unrounded evaluator values, all must hold)

Approved by Badabuba for one Kaggle exploration; deliberately looser than the OPT-02 gate.

- K-a: `mAP(c4) >= 0.2840` (no more than about -0.002 below Model A 0.2860; the literal 0.2840 is used)
- K-b: `AP_small(c4) >= AP_small(A) - 0.005`
- K-c: density bin `>=100`: `mAP_bin(c4) >= mAP_bin(A) - 0.010`

## Outcome

- Gate fails: no CSV, negative result recorded, stop.
- Gate passes: merge the cached A and F **test** predictions with the same c4 rule (no inference rerun), write a lossless
  CSV with `movd.submission`, run `scripts/validate_submission.py` and `scripts/check_submission_independent.py`, record
  its path, size and sha256. Claude never uploads it; the upload decision is the student's.

Passing the gate does **not** make c4 the internal selection over Model A: that still needs the OPT-02 val_full gate
(mAP >= A + 0.003). The CSV is a Kaggle exploration candidate only.
