# CH-10 selection rule (written 2026-11-09 10:10 Kyiv, before any CH-10 variant had finished training)

**Candidates:** the CH-09 baseline `yolo_baseline_marko_3050` (imgsz 1344), `yolo_p2` (imgsz 1344, P2 head) and
`yolo_imgsz1600` (imgsz 1600, approved resolution exception). `yolo_baseline_5070` is a replicate of the baseline
configuration that measures run-to-run / GPU spread. It is reported, but it is not a separate candidate.

**Rule:**

1. Selection metric: **val_fast mAP@[.5:.95] of the final (epoch 6, `last.pt`) checkpoint**, from `movd.evaluate`
   (maxDets 500). That checkpoint is the one evaluated on val_full. No early-epoch checkpoint is picked, because picking
   the best epoch on val_fast and then scoring it would make the comparison uneven between runs.
2. val_full is evaluated for every candidate and reported, but it does not select.
3. If the winner is ahead of the 1344 baseline by less than the measured replicate spread
   (|val_fast mAP, `yolo_baseline_5070` − `yolo_baseline_marko_3050`|), the gain is reported as
   **within run-to-run noise**, and the selection is marked as not significant (one seed).
4. The 1600 run costs about 1.42× the pixels per image. If it wins, the selection record also reports its throughput and
   training time next to the best 1344 run, and states that it changes the protocol resolution. Whether CH-14 may use
   1600 is decided by CH-12's frozen rule and compute budget, not here.
5. Primary anchor-free result for the four-family comparison at the frozen budget (1344): the better of the
   CH-09 baseline and `yolo_p2` under rule 1.
