# Speed recheck: all Ultralytics checkpoints timed back-to-back in one session

Why: the per-run `results/rtdetr_*/speed.json` files were written at the end of each run, hours apart. The same RT-DETR-l
architecture measured 13.8-14.1 FPS for the first three runs (02:15-05:14 Kyiv) but 10.8 FPS for `rtdetr_lr1e4_wb0`
(06:46). Identical architectures cannot differ by that much, so the machine state changed between sessions (laptop on AC,
Windows "Turbo" power plan in both cases; the cause is unknown: likely GPU boost/thermal behaviour).

What: 2026-11-09 ~06:50 Kyiv, the five checkpoints below timed one after another on Nazar's RTX 5070 Laptop GPU with the same
benchmark (`scripts/predict_ultra.py speed`, first 30 val_full frames preloaded, 10 warm-up + 30 timed, batch 1, imgsz 1344,
conf 0.001, max_det 500). The original per-run `speed.json` files are kept unchanged.

| checkpoint | FPS this session | FPS in its own run's speed.json |
|---|---|---|
| rtdetr_baseline | 10.60 | 13.84 |
| rtdetr_q500 | 10.50 | 14.07 |
| rtdetr_lr1e4 | 10.61 | 13.84 |
| rtdetr_lr1e4_wb0 | 10.69 | 10.78 |
| yolo_baseline_marko_3050 (YOLO11s, CH-09 checkpoint) | 64.21 | 72.77 (05:16 session, `../yolo11s_speed_rtx5070.json`); 36.37 on the RTX 3050 |

Use for comparisons: this table compares models within one session. Absolute FPS on this laptop varies by about 20-25%
between sessions, so only same-session ratios (here YOLO11s / RT-DETR-l is about 6x) are stable. 300 vs 500 queries makes
no measurable speed difference. The torchvision Faster R-CNN / RetinaNet FPS come from other sessions and a different timing
scope (GPU-resident tensors), so they are not part of this table.
