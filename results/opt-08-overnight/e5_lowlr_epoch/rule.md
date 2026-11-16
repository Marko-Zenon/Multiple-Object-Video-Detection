# OPT-08 e5_lowlr_epoch: E5 — one extra low-LR epoch from Model A (1 candidate)

Written and committed **before** the CUDA smoke and before any E5 result. Hypothesis, the single intervention,
runtime, start rule and stop rules: [`training_decision.md`](training_decision.md). Configs:
`configs/runs/opt08_e5_lowlr.json` (full) and `configs/runs/opt08_e5_lowlr_smoke.json` (smoke).

Candidate: the epoch checkpoint `outputs/opt08_e5_lowlr/train/weights/epoch0.pt` (sha256 recorded in
`checkpoint.json` right after training), through the E4 raw cache + unchanged CH-13 merge, then x2 (IoU 0.5).
Stages S1-S3 and U1-U8 exactly as TASK.md (S1 is a gate for this single candidate); U7 also needs the GPU recheck of
clip `uav0000305_00000_v`. Comparison baseline: OPT-07 x2.

Frozen parameters (`movd/opt08.py` EXPERIMENTS):

    l1 {"iou": 0.5, "kind": "soup", "tag": "e5l1"}
