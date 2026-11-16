# OPT-12 frozen selection rule (committed before the training produced any result)

Source of truth: the student-approved plan `C:/Users/nazar/.claude/plans/pasted-content-id-58d0-act-as-delegated-cosmos.md`
(sections D and E). Machine-readable constants: `movd/opt12.py` (`CATASTROPHE_*`, `NSTAR_*`, `ALPHAS`, `ALPHA_TIE`,
`WSTAR_SCREEN`, `W070`, `GATES`, `C3_NSTAR_MIN`, `C3_GRID`, `TLR_DEMOTE`, `CLOCK`). Nothing here is loosened after a result.

## Training (`configs/runs/opt12_reg_ft.json`, `scripts/opt12_train.py`)

- RT-DETR-l from `soup_w075.pt` (sha256 `cfaaf0a4…b39d`, c1's weights), train_final (6,072 frames, masked export identical to
  CH-14's), imgsz 1344, batch 1, nbs 16, AMP, seed 42.
- AdamW lr0 1e-4, momentum (beta1) 0.9, weight decay 1e-4, warmup 1 epoch with bias LR from 0, cosine to 0.1 x lr0.
- mosaic 0.5 (closed for the last 2 epochs, 1 if E < 6), scale 0.5, translate 0.1, hsv 0.015 / 0.7 / 0.4, fliplr 0.5.
- Layers 0-1 (HGStem + first HGBlock stage) frozen; every BatchNorm2d in eval mode (statistics frozen at w075's).
- E = clamp(floor((06:15 - start) / measured epoch seconds), 4, 10), fixed from the smoke before the run.
- Monitor: the Ultralytics validator on the val_fast list every epoch; reference R = the same validator on w075.
  Abort only if the monitor after epoch 2 is below R - 0.060, on a non-finite loss, or after a second crash.
- Clock: stop after a saved epoch if the next cannot finish by 06:55; kill at 07:00; no start after 03:45, no resume after 05:30.

## Selection (project evaluator `movd/evaluate.py`, single view, val_fast)

1. Every epoch checkpoint (curve), w075, and AVG3 = mean of the EMA weights of the last 3 completed epochs.
2. N-star = AVG3, unless the last epoch or the best curve epoch is better by >= 0.003.
   N-star is cached (6 views) only if it is >= w075 - 0.010.
3. W(alpha) = (1 - alpha) * w075 + alpha * N-star for alpha in {0.15, 0.25, 0.5}; alpha-star = best, the smallest alpha
   within 0.001 of the best wins. W-star is cached only if it is >= w075 + 0.002.
4. GPU priority: N-star, W-star, then w070 (= 0.70 * Model A epoch0.pt + 0.30 * epoch1.pt).

## Candidates (c1's pipeline unchanged: 6 views at 1344, tile-border drop 0.005, class-wise NMS 0.5, top 500,
five-frame persistence boost IoU 0.5 strength 1). Deltas vs c1, recomputed from its cached predictions.

| Candidate | Construction | Gate (all required) |
|---|---|---|
| C1 | N-star + pipeline | val_full >= +0.0010; width<=1920 >= 0; discovery >= 0; confirmation >= 0; >= 5/7 clips non-negative |
| C2 | W-star + pipeline | val_full >= +0.0005; width<=1920 >= 0; discovery >= 0; confirmation >= 0; >= 5/7 clips |
| C3 | movd/ensemble.py union of c1 and N-star final predictions; grid (c1 w, N w) x NMS IoU in {(1,1),(1,0.75),(0.75,1)} x {0.5,0.6}, chosen by discovery mAP | N-star val_full >= c1 - 0.010; val_full >= +0.0030; width<=1920 >= +0.0010; confirmation >= 0; >= 5/7 clips |
| C4 | w070 + pipeline | val_full >= +0.0005; width<=1920 >= 0; confirmation >= 0; >= 5/7 clips |

Every CSV also needs both validators and a deterministic rebuild with an identical sha256. No gate passed -> no CSV.
Fallback (smoke failure, late start, catastrophe abort, second crash, or no primary candidate passing): c1 and c2 stay the
guaranteed deliverables; C4; salvage W(0.25) of w075 with the best-monitor epoch, same screen and the C2 gate.

## Upload order (sequential test; the agent never uploads)

C1 first. If public(C1) >= 0.2930: C3, then C2. If public(C1) < 0.2930: skip C3; C2 only if alpha-star <= 0.25.
If C1 did not pass or was not built: C2, then C3, then C4. An advisory low-resolution check (T-LR, val clips with
width <= 1920 down-scaled 2x) moves a candidate with delta < -0.005 to the end; it is never a gate.
