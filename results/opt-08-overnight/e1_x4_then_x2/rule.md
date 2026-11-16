# OPT-08 e1_x4_then_x2: E1 — OPT-07 x4 border drop then OPT-07 x2 persistence boost

Written and committed **before** any candidate of this experiment was evaluated on any split. Not changed
after results are seen.

## Hypothesis

x4 (drop tile-view boxes touching an inner tile edge before the CH-13 merge) removes false positives; x2 (boost of
boxes matched in both neighbour frames) promotes temporally consistent boxes. They act on different errors, so the
combination may beat x2 alone. Both rules are used exactly as frozen in OPT-07; nothing is retuned.

## Exact deterministic rule

1. OPT-03 e2 / OPT-07 x4 on the Model A raw per-view cache (`scripts/opt03_edge.py::run_candidate(split, 0.0, 0.005)`,
   unchanged); the output must equal the stored OPT-07 x4 predictions exactly (checked every run, else stop).
2. OPT-07 x2 (`movd.opt07.persistence_boost`, IoU 0.5) on that output. No box moved, added or removed by step 2.

Single candidate (no grid), so S1 is a gate, not a selection.

Frozen parameters (`movd/opt08.py` EXPERIMENTS, checked by `scripts/opt08_loop.py screen`):

    c1 {"factor": 0.0, "iou": 0.5, "kind": "x4_then_x2", "margin": 0.005}

Comparison baseline: OPT-07 x2 (val_full predictions `6c3a929f…2c03`, public 0.2805168313).

## Stages and gate

Fixed in [`../TASK.md`](../TASK.md) before this rule: S1 discovery screen of every listed candidate (4 clips only; the
selection is the highest discovery mAP, ties to the earlier candidate; S1 passes if its delta vs x2 >= +0.0003),
S2 the selection once on the 3 confirmation clips (delta vs x2 >= 0), S3 val_full once only after S1 and S2 pass,
judged by U1-U8 against x2. Rejected if any stage or gate check fails; never retuned after a result.
`python scripts/opt08_loop.py screen|confirm|full|determinism|test <experiment>`.
