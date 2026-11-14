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

## Results

See `HANDOFF.md` (generated from the stored records) for the numbers, the gate outcomes, the CSV paths and hashes and
the recommended upload order.

## TODO for the students (report prose is yours)

- TODO: Why did flip-only fine-tuning over-fit (OPT-10 D2), and what did augmentation + frozen BN change in the curve?
- TODO: Interpret the gate outcomes of C1-C4 against the transfer flags (width<=1920, confirmation, per clip).
- TODO: After uploading, record the public scores and compare them with the val deltas.

Cite: Ultralytics RT-DETR (AGPL-3.0), Zhao et al. 2024 (RT-DETR), Wortsman et al. 2022 (model soups / WiSE-FT).
