# OPT-08 e3_broader_support: E3 — broader conservative temporal support (4 frozen candidates)

Written and committed **before** any candidate of this experiment was evaluated on any split. Not changed
after results are seen.

## Hypothesis

x2 needs a match in exactly t-1 and t+1. A detector miss in one neighbour frame, or the first / last frame of a clip,
blocks the boost. Allowing support from t-2 / t+2 (one axis: which frames may support a box) may extend the gain
without adding false boxes.

## Exact deterministic rule (`movd.opt08.broader_support`)

Input: Model A cached CH-13-pipeline predictions (as E2). For each box, the best same-class IoU match (first maximum)
in t-2, t-1, t+1, t+2 of the same clip; a match needs IoU >= 0.5. score := max(score, mean(own score, supporting
scores)); never lowered, no box moved, added or removed. Modes:
* `endpoint_one_sided` (b1): x2, plus at a clip end (t-1 or t+1 missing) both frames on the existing side matched.
* `gap_tolerant` (b2): each side supported by its t-+1 match, else its t-+2 match; both sides needed; 3-score mean.
* `five_frame_mean` (b3): x2 condition; the mean also includes matched t-2 / t+2 scores.
* `gap_tolerant_endpoint` (b4): b2 plus b1's clip-end rule.
No interpolation, no new boxes.

Frozen parameters (`movd/opt08.py` EXPERIMENTS, checked by `scripts/opt08_loop.py screen`):

    b1 {"kind": "support", "mode": "endpoint_one_sided"}
    b2 {"kind": "support", "mode": "gap_tolerant"}
    b3 {"kind": "support", "mode": "five_frame_mean"}
    b4 {"kind": "support", "mode": "gap_tolerant_endpoint"}

Comparison baseline: OPT-07 x2 (val_full predictions `6c3a929f…2c03`, public 0.2805168313).

## Stages and gate

Fixed in [`../TASK.md`](../TASK.md) before this rule: S1 discovery screen of every listed candidate (4 clips only; the
selection is the highest discovery mAP, ties to the earlier candidate; S1 passes if its delta vs x2 >= +0.0003),
S2 the selection once on the 3 confirmation clips (delta vs x2 >= 0), S3 val_full once only after S1 and S2 pass,
judged by U1-U8 against x2. Rejected if any stage or gate check fails; never retuned after a result.
`python scripts/opt08_loop.py screen|confirm|full|determinism|test <experiment>`.
