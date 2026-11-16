# OPT-08 E5 — one extra low-LR epoch from Model A (training decision, frozen before the CUDA smoke)

## Evidence and hypothesis

`results/ch-14-final/epochs.csv` (CH-14 final_rtdetr, project evaluator, val_fast single-model): epoch 1 (Model A)
0.2790, epoch 2 0.2585, epoch 3 0.2614, epoch 4 0.2609, epoch 5 0.2531, epoch 6 0.2381, while the training losses fall
every epoch (giou 0.749 -> 0.390). The Ultralytics LR (pg0) was still warming up: 2.78e-4 at the end of epoch 1,
4.64e-4 at the end of epoch 2 (`optimizer=auto` -> AdamW, lr0 8.33e-4, 3-epoch warmup). So the post-epoch-1 drop
comes with a rising LR and falling train loss (fitting the 6,072 train frames, losing val).
**Hypothesis:** one more epoch from Model A at a low constant LR consolidates without the high-LR drift, so its
CH-13 + x2 predictions beat x2.

## The single intervention

`configs/runs/opt08_e5_lowlr.json` = Model A's recipe (`configs/runs/final_rtdetr.json`: train_final 6,072 frames,
imgsz 1344, batch 1, AMP, seed 42, flip-only, 300 queries, masked export / frozen ignore policy) started from Model A's
weights (`MOVD-ch14/.../epoch0.pt`, sha256 `6ff4a430…5b0b`, a new run, not a resume), with one change: AdamW at a
constant lr0 = 5e-5 (18 % of Model A's last LR, 6 % of the auto lr0). `overrides.epochs = 1` (exactly one extra epoch;
Ultralytics then uses no warmup and a constant LR). Optimizer state is new (Model A's checkpoint has none to reuse
in a non-resume run). Nothing else changes (no augmentation, regularisation, data or architecture change).

## Order, checks and stop rules

1. Starts only after E4 has finished or been rejected (one GPU process), and only if
   start time + 110 min (smoke 6, epoch ~27, discovery 11, confirmation 13, test 31, margin) <= 06:50 Kyiv,
   i.e. a start before about 05:00; never after 06:15. Otherwise E5 is recorded as not run (time).
2. CUDA smoke (`opt08_e5_lowlr_smoke`, ~100 frames, `--stop-after-epoch 1`, then `--resume`): finite losses
   (train_ultra raises on a non-finite loss), checkpoint written, resume completes epoch 2, the checkpoint reloads and
   runs one CH-13 raw pass, and its weights are much closer to Model A than to COCO `rtdetr-l.pt` (proves the start
   point). Any failure: E5 rejected, no full run.
3. Full epoch: `python scripts/train_ultra.py --config configs/runs/opt08_e5_lowlr.json --device 0`. Stop rule: a
   non-finite loss, or a projected end later than 05:45 Kyiv from the first 300 batches, stops it (logs kept, E5
   rejected). The epoch checkpoint `outputs/opt08_e5_lowlr/train/weights/epoch0.pt` is the candidate.
4. Evaluation exactly as E4 (one candidate `l1`): raw 6-view cache on the discovery clips, CH-13 merge, x2;
   S1 discovery >= x2 + 0.0003; S2 confirmation >= x2; S3 val_full, U1-U8; U7 includes a GPU re-inference of clip
   `uav0000305_00000_v` equal to its cached raw shard; test cache + CSV only if eligible.

## Pre-result corrections (after the CUDA smoke, before the full run; no E5 metric existed)

* The smoke log showed `AdamW(lr=5e-05, momentum=0.937)`: an explicit optimizer takes Ultralytics' default beta1
  0.937, while Model A's `optimizer=auto` used 0.9. `overrides.momentum = 0.9` restores Model A's value, so the
  learning rate remains the only change. (The smoke config keeps 0.937; the smoke only checks mechanics.)
* Smoke results (`smoke_checks.json`, `results/opt08_e5_lowlr_smoke/`): finite losses (0.41 / 0.45 / 0.07 at start,
  i.e. Model A level, far below a COCO start), checkpoint written, resume completed epoch 2, the checkpoint reloads
  and runs a CH-13 raw pass; learned parameters equal Model A's exactly (L2 0 vs 14.05 to COCO): start point proven,
  but the ~3 accumulated optimizer steps of the smoke were all skipped (AMP GradScaler warm-up). Added stop rule:
  after the full epoch, if the learned parameters still equal Model A's (L2 = 0), E5 is rejected as "no update".
* Throughput 4.4 it/s, so the epoch is ~23 min (6,072 frames).
