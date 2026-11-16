# OPT-04 selection rule (committed before any OPT-04 result)

OPT-04 is an optional post-plan experiment, requested and approved by student Badabuba on 2026-11-10
(plan file `pasted-content-id-afd9-create-a-quirky-raven.md`; its §7 configs and §8 gates are copied here).
Worker: Claude. Claude uploads nothing to Kaggle; the uploads and the two final Kaggle selections stay with the students.
Gates are never relaxed after a result is seen. No third experiment is run under any outcome.

When this file was committed, no OPT-04 checkpoint, prediction or metric existed. The only numbers in it are the
Model A and fallback-F values already registered on `develop`.

## References (registered, unchanged)

**Model A** = CH-14 `final_rtdetr` `epoch0.pt` (epoch 1 of 6, inside the 3-epoch warmup), read in place from
`C:/UCU/AI/hw2/MOVD-ch14/outputs/final_rtdetr/train/weights/epoch0.pt`, sha256
`6ff4a430188405b5d13356872c63f141c4d3dbc7b747983cdf733ce386aa5b0b`, + the CH-13 pipeline
(`configs/runs/final_inference.json`, sha256 `2e730aee…4bd7`). Evidence: `results/ch-14b-rtdetr-short-finetune/stageA_epoch1/`.

| Split | mAP | AP50 | AP75 | AP_S | AP_M | AP_L | 0-19 | 20-49 | 50-99 | ≥100 |
|---|---|---|---|---|---|---|---|---|---|---|
| val_fast (714) | 0.2892 | — | — | 0.1427 | — | — | 0.2132 | 0.3018 | 0.3103 | 0.3515 |
| val_full (2,846) | 0.2860 | 0.5358 | 0.2600 | 0.1504 | 0.2568 | 0.4812 | 0.2081 | 0.2981 | 0.3073 | 0.3497 |

Model A's test-like-subset value (val_fast frames with width ≤ 1920) is computed by `scripts/opt04_gates.py` with
`movd.evaluate.evaluate(image_ids=…)` from Model A's stored val_fast pipeline predictions
(`C:/UCU/AI/hw2/MOVD-ch14b/outputs/ch14b/stageA_epoch1/val_fast/pipeline_predictions.json`, sha256 `87137d7d…5a0e`,
checked before use). Model A CSV `e587c68b…0da7` (public 0.2774301362). F CSV `73f73e21…793b` (val_full 0.2696, public 0.2328849249).

All mAP values come from the project evaluator (`movd/evaluate.py`, maxDets 500, frozen crowd/ignore policy).
Each metric is compared unrounded against the literal threshold below. A "lead" in a secondary check means X > A (both unrounded).

## Candidates

**P (primary): RT-DETR-x, Model A recipe, epoch 1.** Exactly one variable changes relative to Model A: the pretrained model,
`rtdetr-l.pt` → the official Ultralytics COCO `rtdetr-x.pt` (sha256 recorded on download; it must load as an
`RTDETRDecoder` model with 300 queries and 80 COCO classes). Everything else is identical to `configs/runs/final_rtdetr.json`,
including `epochs: 6`. The schedule length sets the warmup length (3 epochs = 18,216 iterations), so the epoch-1 LR,
momentum and accumulation trajectory equal Model A's. Training stops right after the epoch-1 checkpoint `epoch0.pt` is written
(`stop_after_epoch: 1`). **No second epoch is run.** Config `configs/runs/opt04_rtdetrx_e1.json`. Inference uses
`configs/runs/final_inference.json` unchanged (6 passes at 1344).

**FB (fallback): RT-DETR-l at imgsz 1600, Model A recipe, epoch 1.** One conceptual variable changes relative to Model A:
the input size 1344 → 1600, for training (`overrides.imgsz: 1600`) and for all 6 inference passes
(`configs/runs/opt04_inference_1600.json` is a copy of `final_inference.json` whose only change is `imgsz: 1600` in each
pass. Its descriptive text still says 1344, because nothing else was changed). The pipeline is otherwise identical: same views,
tile fraction 0.6, NMS 0.5, top 500. It starts from COCO `rtdetr-l.pt`
(sha256 `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f`), Model A's own init, not from Model A.
This uses the pre-approved resolution exception (plan.md §11, approved by Badabuba), which mirrors CH-10's.
Config `configs/runs/opt04_rtdetrl1600_e1.json`.

Both use train_final (6,072 frames, manifest checked by `scripts/train_ultra.py`, split sha as in `results/final_rtdetr/config.json`),
the masked export (frozen ignore policy), batch 1, nbs 64 (accumulation ramps 1→64 over the warmup), workers 2,
optimizer auto (AdamW lr0 0.000833, β1 0.9 with momentum warmup 0.8→0.9, wd 0.0005, warmup_bias_lr→0.0), lrf 0.01,
linear lf, warmup_epochs 3, no frozen layers, flip-only augmentation (`fliplr 0.5`; mosaic, mixup, scale, HSV and translate 0;
close_mosaic 0), seed 42, deterministic, AMP, 300 queries, max_det 500, Ultralytics validation off.
Checkpoint selection: the single epoch-1 checkpoint `epoch0.pt` (no choice).
Smoke configs `opt04_rtdetrx_smoke.json` / `opt04_rtdetrl1600_smoke.json`: as the run, epochs 2,
`smoke_fraction 0.0165` (~100 frames), `--stop-after-epoch 1`, then `--resume`. Smoke numbers are never results.

Resume: checkpoints are written at epoch end only, so a crash inside epoch 1 cannot be resumed. The run is restarted once as
`<run_id>_r2` (same config and seed); the failed directory is kept and marked FAILED. A second failure triggers S2.

## Gates

| Gate | Where | Pass condition | On fail |
|---|---|---|---|
| G0 smoke / feasibility | ~100 iterations | finite losses; stop-after-epoch + resume work; **peak reserved ≤ 7.0 GB** (l: smoke peak = full-run peak); no OOM / auto-batch-reduce log line; projected epoch ≤ 120 min; 30-frame pipeline speed projects test ≤ 90 min | S1 → fallback (the model is never changed to make it fit) |
| G1 training identity | after epoch 1 | `epoch0.pt` written; finite losses; train_args epochs 6, imgsz (1344 for P, 1600 for FB), batch 1, seed 42, fraction 1.0, train_final; no OOM-retry event; at most one restart | S2 → fallback |
| G2 val_fast screen | pipeline, 714 frames | mAP ≥ **0.2922** (A + 0.003); AP_S ≥ **0.1397**; ≥100 bin ≥ **0.3465**; test-like subset (width ≤ 1920) mAP ≥ Model A's on the same subset | S3 → stop; fallback (if this was P) |
| G3 val_full → test CSV | pipeline, 2,846 frames, **run once** | mAP ≥ **0.2890** (A + 0.003); AP_S ≥ **0.1474**; ≥100 bin ≥ **0.3447**; ≥ 5 of 9 secondary checks vs A (AP50, AP75, AP_S, AP_M, AP_L, 4 density bins) | S4 → no CSV; fallback (if P) |
| G4 upload-ready | test run | non-smoke; 3,811 rows; `validate_submission.py` and `check_submission_independent.py` exit 0; 0 rows differ from the raw predictions; test time recorded; sha256 recorded; CSV differs from `e587…`/`73f7…` | fix the pipeline, never the gate |
| G5 primary replacement (rule R) | val_full | mAP ≥ **0.2923** (A + 0.0063, replicate spread) and G3's guards | for P: **S6 → run FB** (if before its cutoff); P's CSV (if G4) stays upload-ready |
| S5 deadline | — | P must reach G2 by Thu 12 Nov 12:00 Kyiv | → fallback if FB can still start by its cutoff |
| S6 marginal P | — | P passes G3 but fails G5 | → run FB if it can start by Thu 12 Nov 18:00 Kyiv; if P passes G5, FB is never run |

Stop conditions: S1 = P fails G0. S2 = P fails G1 (after at most one restart). S3 = P fails G2. S4 = P fails G3.
S5 = P misses its G2 deadline (Thu 12 Nov 12:00 Kyiv). **S6 = P passes G3 but fails G5.** On any of S1–S6, FB runs
if it can start by the frozen **Thu 12 Nov 18:00 Kyiv** cutoff; otherwise "FB not run: deadline" is recorded.
**FB is never run if P passes G5.** FB goes through the same gates G0–G5 against Model A, with the same thresholds.

Hard cutoffs: no training starts after Fri 13 Nov 12:00 Kyiv; no test inference starts after Fri 13 Nov 18:00 Kyiv.

### P vs FB (only if both pass G3 and G4)

On the frozen val_full, the higher val_full pipeline mAP wins. If |Δ| < 0.0063, the winner is the one that leads on
more of the 9 secondary checks. If still tied, P wins (protocol-native 1344, no exception). **Never decided by the public
LB.** The loser stays a validated but unselected CSV.

### Kaggle public score

The public score never promotes a candidate. It is an anomaly veto only: public < 0.2329 (F) → investigate the format and pipeline,
and do not select X until it is explained.

## Final two selections (plan §10)

| Outcome | Primary | Secondary |
|---|---|---|
| Neither P nor FB passes G3/G4 (current default) | **Model A** `e587c68b…0da7` (val_full 0.2860, public 0.2774301362) | **F** `73f73e21…793b` (val_full 0.2696, public 0.2328849249) |
| One candidate passes G4, not G5 | Model A | that candidate (replaces F: a better and more diverse hedge) |
| One candidate passes G4 **and** G5 | that candidate | Model A |
| Both pass G4 | stronger one (by the frozen "P vs FB" rule) as primary if it passes G5, else Model A | the other of {stronger candidate, Model A} |
| Never | `val_roundtrip` (`a411f7fb…`), FRCNN 0.2012, any CSV that failed G4, a candidate under the public-score anomaly veto | |

At most two new upload-ready CSVs (one from P, one from FB), each only if it passes G4 on its own. Nazar decides which to upload,
at most once each, and re-uploads a file only for a technical Kaggle error (≤ 2 per file). No configuration is changed
in response to a Kaggle score.

## Always reported, never gating

Per-class AP, single-pass numbers, AP50/AP75/AP_M/AP_L, the same-session speed of A vs X (`opt04_infer.py speed`),
peak memory, wall-clock.

## Known biases

* val_fast (714 frames) is a subset of val_full (2,846 frames).
* One seed per candidate. G5 uses the 0.0063 replicate spread as the noise unit.
* val contains a 3840-wide clip that test does not have. The width ≤ 1920 subset guard in G2 addresses this partly.
* FB's resolution gain may partly overlap with the tile zoom of the CH-13 pipeline, and FB is not protocol-comparable at 1344.
