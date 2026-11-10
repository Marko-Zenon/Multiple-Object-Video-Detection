# OPT-02 — RT-DETR-l two-model ensemble (optional post-plan experiment): negative result

**Outcome: no candidate passes the predeclared val_fast gate, so model selection stops. Model A (CH-14b epoch-1
checkpoint + frozen CH-13 inference, val_full 0.2860, Kaggle public 0.2774301362) stays the selected model.
No val_full ensemble evaluation, no new test CSV, no Kaggle upload.**

Rule, grid and gates: [`selection_rule.md`](selection_rule.md) and [`candidates.json`](candidates.json), committed in
`06610a5` before any OPT-02 metric. Worker: Claude for Badabuba, 2026-11-10. CPU only; the RTX 5070 was not used.

## Inputs

- Model A = CH-14 `final_rtdetr` `epoch0.pt` (sha256 `6ff4a430…5b0b`), Model B = CH-11 `rtdetr_baseline` `last.pt`
  (sha256 `da5d3625…e872`), both with the frozen CH-13 pipeline (`configs/runs/final_inference.json`).
- Cached predictions only. They are the CH-13 pipeline output **after** the per-frame top-500 cap; no pre-cap
  predictions exist (limitation stated in the rule before testing). On val_fast, 599/714 Model A frames and 711/714
  Model B frames are already at 500.
- [`identity.json`](identity.json): both checkpoint hashes and every input with a committed hash match; the cached
  val_fast / val_full predictions of both models reproduce their registered metrics **exactly** (A 0.2892 / 0.2860,
  B 0.2725 / 0.2696; metrics, per-class and density dicts identical).

## val_fast screening (714 frames, project evaluator, maxDets 500)

Gate: mAP >= A + 0.003, AP_small >= A - 0.003, `>=100` bin >= A - 0.005, projected test time <= 90 min.

| candidate | w_A:w_B | NMS | mAP | Δ mAP | AP50 | AP75 | AP_S | AP_M | AP_L | `>=100` | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A (baseline) | — | — | 0.2892 | — | 0.5397 | 0.2663 | 0.1427 | 0.2594 | 0.4901 | 0.3515 | — |
| c1_w100_nms05 | 1:1 | 0.5 | 0.2899 | +0.0007 | 0.5393 | 0.2663 | 0.1487 | 0.2545 | 0.4802 | 0.3463 | fail (G-a, G-c) |
| c2_w100_nms06 | 1:1 | 0.6 | 0.2904 | +0.0012 | 0.5402 | 0.2668 | 0.1497 | 0.2550 | 0.4804 | 0.3468 | fail (G-a) |
| c3_w075_nms05 | 1:0.75 | 0.5 | 0.2918 | +0.0026 | 0.5442 | 0.2690 | 0.1437 | 0.2583 | 0.4926 | 0.3520 | fail (G-a) |
| c4_w075_nms06 | 1:0.75 | 0.6 | 0.2921 | +0.0029 | 0.5443 | 0.2694 | 0.1441 | 0.2587 | 0.4931 | 0.3523 | fail (G-a) |
| c5_w050_nms05 | 1:0.5 | 0.5 | 0.2901 | +0.0009 | 0.5417 | 0.2670 | 0.1430 | 0.2588 | 0.4908 | 0.3516 | fail (G-a) |
| c6_w050_nms06 | 1:0.5 | 0.6 | 0.2904 | +0.0012 | 0.5420 | 0.2670 | 0.1431 | 0.2591 | 0.4911 | 0.3520 | fail (G-a) |
| c7_specialist_bicycle | bicycle B, rest A | 0.5 | 0.2902 | +0.0010 | 0.5372 | 0.2710 | 0.1429 | 0.2598 | 0.4920 | 0.3531 | fail (G-a) |

Per-class AP (val_fast):

| candidate | person | bicycle | car | van | truck | tricycle | bus | motor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (baseline) | 0.3000 | 0.2534 | 0.4535 | 0.3297 | 0.3126 | 0.2569 | 0.2537 | 0.1540 |
| c1_w100_nms05 | 0.2894 | 0.2664 | 0.4395 | 0.3283 | 0.3320 | 0.2609 | 0.2617 | 0.1408 |
| c2_w100_nms06 | 0.2904 | 0.2679 | 0.4406 | 0.3283 | 0.3319 | 0.2612 | 0.2612 | 0.1414 |
| c3_w075_nms05 | 0.3002 | 0.2559 | 0.4590 | 0.3292 | 0.3139 | 0.2649 | 0.2571 | 0.1543 |
| c4_w075_nms06 | 0.3006 | 0.2566 | 0.4600 | 0.3294 | 0.3138 | 0.2648 | 0.2572 | 0.1544 |
| c5_w050_nms05 | 0.3003 | 0.2544 | 0.4581 | 0.3298 | 0.3129 | 0.2583 | 0.2532 | 0.1541 |
| c6_w050_nms06 | 0.3007 | 0.2551 | 0.4586 | 0.3299 | 0.3129 | 0.2586 | 0.2530 | 0.1542 |
| c7_specialist_bicycle | 0.3000 | 0.2614 | 0.4535 | 0.3297 | 0.3126 | 0.2569 | 0.2537 | 0.1540 |

Density bins (mAP by target objects per frame) and detection counts (val_fast):

| candidate | 0-19 | 20-49 | 50-99 | `>=100` | detections | mean / frame | frames at 500 | share from B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (baseline) | 0.2132 | 0.3018 | 0.3103 | 0.3515 | 352,886 | 494.24 | 599 / 714 | — |
| c1_w100_nms05 | 0.2170 | 0.3093 | 0.3123 | 0.3463 | 357,000 | 500.00 | 714 / 714 | 0.2500 |
| c2_w100_nms06 | 0.2171 | 0.3098 | 0.3129 | 0.3468 | 357,000 | 500.00 | 714 / 714 | 0.2541 |
| c3_w075_nms05 | 0.2200 | 0.3056 | 0.3134 | 0.3520 | 357,000 | 500.00 | 714 / 714 | 0.1262 |
| c4_w075_nms06 | 0.2199 | 0.3060 | 0.3137 | 0.3523 | 357,000 | 500.00 | 714 / 714 | 0.1367 |
| c5_w050_nms05 | 0.2167 | 0.3034 | 0.3109 | 0.3516 | 357,000 | 500.00 | 714 / 714 | 0.0616 |
| c6_w050_nms06 | 0.2167 | 0.3035 | 0.3114 | 0.3520 | 357,000 | 500.00 | 714 / 714 | 0.0714 |
| c7_specialist_bicycle | 0.2132 | 0.3064 | 0.3088 | 0.3531 | 352,768 | 494.07 | 518 / 714 | 0.0336 |

Full-precision values: [`screening_val_fast.csv`](screening_val_fast.csv), [`selection_val_fast.json`](selection_val_fast.json)
and one `val_fast/<candidate>/{metrics.json,per_class.csv}` per row (incl. the baseline `val_fast/A/`).

## Decision

`selection_val_fast.json`: `passing = []`, `chosen = null`. Every candidate fails G-a (best: c4 +0.0029 < +0.003);
c1 also fails G-c (`>=100` bin -0.0052). By the rule: model selection stops, Model A and its existing
0.2774301362 submission are kept, `confirm` (val_full) and `test` (CSV) are not run. The rule is not relaxed.

## Runtime and memory

- Merge (CPU, cached predictions): 1.0-1.5 s for the 714 val_fast frames per candidate, about 6 s projected for 3,811
  test frames.
- Projected test time if both models had to be run fresh (gate G-d): A 1,729.5 s (CH-14b test run) + B 1,699.7 s
  (CH-13 test run, `MOVD-ch13/outputs/ch13_test/run.json`) + merge = **about 57.3 min** for every candidate (the
  specialist needs both models too), within the 90-min limit. Only G-a failed.
- Peak process memory of the screen (incl. pycocotools on 8 prediction sets): 1.94 GB (`peak_wset`). No GPU memory
  used (the CH-13 pipeline itself peaks at 0.70 GB allocated per model on the RTX 5070).

## Determinism and regeneration

[`determinism.json`](determinism.json): merging the same inputs twice gives the same sha256; re-merging and
re-evaluating every candidate reproduces each tracked `val_fast/<candidate>/metrics.json` exactly. Separately,
the screen was run three times (see deviations); all eight `metrics.json` files were byte-identical across runs.

## Deviations and notes

1. The first screen run computed and wrote all eight `metrics.json` files, then crashed while writing
   `screening_val_fast.csv` (the CSV writer took its columns from the baseline row only). After the fix the screen was
   re-run unchanged; the metrics were byte-identical to the first run (kept outside Git in `outputs/opt02/`), so no
   result was changed or re-selected.
2. Inputs are capped (see Inputs); merging uncapped lists might differ slightly. Not tested (would need fresh GPU
   inference for both models; not required to answer the question under the student's no-GPU instruction).
3. Screening on val_fast only, as the student requested; val_fast is a subset of val_full.

## TODO for the students (report prose is yours)

- TODO: interpret why equal weighting raises AP_small, bicycle and truck but lowers person, car, motor, AP_L and the
  densest bin, while down-weighting B (0.75) gives a small, uniform gain that stays under the margin.
- TODO: decide whether to mention OPT-02 in the report as a documented negative ensemble result.
