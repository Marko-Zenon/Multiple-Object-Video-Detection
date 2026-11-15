# OPT-12 — final-night regularised fine-tune (evidence)

Optional post-plan chunk, student-approved 2026-11-15 (plan `pasted-content-id-58d0-act-as-delegated-cosmos.md`).
Frozen rule: [`selection_rule.md`](selection_rule.md) (committed `a1bc6a4`, before any training result).
The agent never uploads; public scores are added to `results/submissions.csv` only after a student reports them.

## What was run

- **Training** (`configs/runs/opt12_reg_ft.json`, `scripts/opt12_train.py`, evidence `results/opt12_reg_ft/`):
  RT-DETR-l from soup w075 (c1's weights), train_final 6,072 frames, imgsz 1344, batch 1 / nbs 16, AdamW 1e-4 cosine,
  mosaic 0.5 / scale 0.5 / translate 0.1 / HSV / flip, layers 0-1 frozen, BatchNorm statistics frozen, 10 epochs.
- **Smoke** (`results/opt12_reg_ft_smoke/`): start weights equal soup_w075 tensor for tensor (`start_identity.json`),
  BN running statistics unchanged through train, stop and resume (`observations.json`), augmented samples
  `figures/augmented_samples.jpg`, peak 3.5 GB reserved, 4.3 it/s.
- **Morning** (`scripts/opt12_morning.py`): c1 rebuilt exactly from its caches (`baseline_c1.json`), single-view
  selection (`selection.json`), candidate gates (`cand/<tag>/gate.json`), CSVs and validators (`test/<C>/`),
  advisory low-resolution check (`tlr.json`), hand-off (`HANDOFF.md`).
- **Runner** (`scripts/opt12_night.py`), started outside the Claude session (`scripts/opt12_launch.ps1`, WMI);
  state `outputs/opt12/state.json`, events `outputs/opt12/logs/events.jsonl` (git-ignored).

## Results (all numbers from movd/evaluate.py; full tables in `HANDOFF.md`)

- Training completed 10 epochs (00:23-04:29 Kyiv, 246 min, 3.49 GB peak), BN statistics frozen throughout.
- Single-view val_fast: w075 0.2882; best epoch 3 0.2916; AVG3 (epochs 8-10) 0.2725 -> N-star = epoch 3 (frozen rule).
  WiSE-FT W(alpha) = (1 - alpha) w075 + alpha N-star: 0.3006 / 0.3032 / 0.3045 for alpha 0.15 / 0.25 / 0.5 -> alpha-star 0.5.
- Gates vs c1 (val_full, width<=1920, discovery, confirmation, clips >= 0):
  C1 N-star +0.0066 / -0.0003 / +0.0034 / -0.0351 / 2 of 7 - rejected;
  C2 W-star +0.0148 / +0.0137 / +0.0133 / -0.0267 / 5 of 7 - rejected by the confirmation check only;
  C3 ensemble(c1, N-star) +0.0090 / +0.0029 / +0.0049 / -0.0308 / 4 of 7 - rejected;
  C4 w070 +0.00004 / +0.00001 / -0.0003 / +0.0003 / 3 of 7 - rejected.
- C2 exception: the student (Badabuba) granted a one-off, documented exception at 10:21 Kyiv
  (`cand/wstar/exception.json`; `cand/wstar/gate.json` still records `passed: false`). C2's only losses are the two widest
  val clips (117, 2720 px, -0.028; 268, 3840 px, -0.030); all four test-width clips gain. CSV `opt12_c2_wiseft_test.csv`,
  sha256 `9e2ed403…5fb4`, both validators pass, fresh-process rebuild identical.
- Advisory T-LR (width<=1920 val clips down-scaled 2x): C1 +0.025, C2 +0.048, C3 +0.028 vs c1.
- Protected artifacts re-hashed at start and end: all unchanged (`protected_start.json`, `protected_end.json`).

Data note: `results/submissions.csv` lists c1's val_full as 0.29739905, while the stored OPT-09 metric
(`results/opt-09-e6-tuning/c_final/c1/val_full/metrics.json`) and the OPT-12 rebuild both give 0.2973993546311055;
the CSV value is a transcription slip in the 7th decimal. OPT-12 gates use the stored metric.

## TODO for the students (report prose is yours)

- TODO: Why did flip-only fine-tuning over-fit (OPT-10 D2), and what did augmentation + frozen BN change in the curve?
- TODO: Interpret the gate outcomes of C1-C4 against the transfer flags (width<=1920, confirmation, per clip).
- TODO: After uploading, record the public scores and compare them with the val deltas.

Cite: Ultralytics RT-DETR (AGPL-3.0), Zhao et al. 2024 (RT-DETR), Wortsman et al. 2022 (model soups / WiSE-FT).
