# OPT-03 — diagnosis of Model A (val_fast only)

Technical evidence note (worker: Claude, for Badabuba). Not report prose. Every number below is in
`diagnosis/` and can be regenerated with `scripts/opt03_diagnose.py {identity,ablations,tables}`.

## Inputs and identity

* Model A = CH-14 `epoch0.pt` (sha256 `6ff4a430…5b0b`) + frozen CH-13 pipeline (`configs/runs/final_inference.json`:
  full + flip + 2×2 tiles at 1344, decoder top-300 per view, conf 0.001, class-wise NMS 0.5, global top 500).
* New raw per-view cache of Model A (`scripts/opt03_cache.py`, 6 views × 300 queries × 8 class scores per frame,
  ignored `outputs/opt03/raw/`). Re-merged on the CPU it reproduces the stored CH-14b val_fast predictions
  **exactly** (352,886 boxes, max abs difference 0.0, mAP 0.28922; `diagnosis/identity.json`).
* Only val_fast (714 frames) was used for the diagnosis. val_full was cached but not evaluated.
* mAP numbers come from `movd/evaluate.py` (maxDets 500). FP / FN tables use a simple diagnostic matcher
  (`movd/opt03.py::match_frame`, IoU 0.5, greedy by score, crowd regions ignored by intersection / box area).

## Findings

1. **The 500 cap is not the bottleneck.** 599 / 714 frames reach 500 boxes (768 boxes per frame after NMS on
   average), but removing the cap entirely (oracle, not scoreable on Kaggle) gives mAP 0.28928 vs 0.28922 (+0.00006).
   Only 78 of 28,685 target GT boxes (0.27 %) are found below rank 500. The median score at rank 500 is 0.058; 99.8 % of
   the scored TPs are above 0.05. A better top-500 allocation or class-wise score calibration can only change which
   boxes survive the cap (COCO AP is per class, so a monotone per-class rescaling changes nothing else), so its
   ceiling is about +0.0001. **Rejected.**
2. **Tiles help small objects but add many false positives.** View ablations (`diagnosis/ablations_val_fast.csv`):

   | views | mAP | AP_S | ≥100 bin |
   |---|---:|---:|---:|
   | full only | 0.2775 | 0.1069 | 0.3656 |
   | full + flip | 0.2867 | 0.1052 | 0.3681 |
   | tiles only | 0.2475 | 0.1536 | 0.3420 |
   | all 6 (Model A) | 0.2892 | 0.1427 | 0.3515 |
   | all but tile2 / all but tile3 | 0.2901 / 0.2899 | 0.1423 / 0.1428 | 0.3491 / 0.3555 |

   Tiles add +0.037 AP_S but cost −0.017 on the densest bin, and removing some single tiles *raises* mAP.
   Per scored box, tile views have precision@0.5 of 5–8 % vs 9–10 % for the full and flip views.
3. **The tile FPs sit on the tile borders.** A tile box that touches an inner edge of its tile (an edge that is not a
   frame edge) is a cut-off part of an object that continues in the neighbouring area. In the scored set there are
   **47,158 such boxes (13 % of all scored boxes), with precision@0.5 of 0.35 % (162 TPs)**, mean score 0.091, versus
   7.6 % precision for the other 180,086 tile boxes (`tables.json` → `tile_inner_edge`). Their score (0.09) is above
   the median FP score (0.079), so inside each class they rank above many low-score TPs and lower precision along the
   PR curve. Class-wise NMS at 0.5 does not remove them, because a half box overlaps the full box by IoU ≈ 0.5 or less.
4. **Errors by class / size / density / resolution / occlusion** (recall@0.5 of the scored set, `recall_by_*`):
   heavy occlusion 0.38 (vs 0.92 none), small 0.70 (vs 0.96 large), bus 0.55, truck 0.75, motor 0.78. FPs are
   dominated by background boxes, then wrong-class boxes (truck 5,452 of 15,585 scored truck boxes are on another
   class's object; bus 1,386 of 2,047). The worst bin is the 3840×2160 clip (recall 0.58; it is the whole 0–19 density
   bin). **The test set has no frame wider than 1920** (sizes 680–1920 px), so a fix aimed at 4K frames would not
   transfer. Calibration differs by class (precision at score 0.10–0.20: person 2.4 %, motor 1.1 %, truck 0.1 %), but
   by finding 1 this only matters through the cap.
5. **Training state.** The CH-14 curve (`results/final_rtdetr/curve.json`) is best at epoch 1 (single pass val_fast
   0.2790) and falls to 0.2381 at epoch 6 while the train losses keep falling (giou 0.749 → 0.390). Epoch 1 is the best
   point of this run; later epochs overfit the 6,072 strongly correlated frames (every 4th frame of each clip). A short
   fine-tune or more epochs is not supported by this evidence; RT-DETR-x is not supported either (recall@0.5 is already
   83 %, the loss is in precision, and a new x run would not fit the remaining time before the 23:55 close).
6. **Temporal consistency.** 5,038 GT are missed; 2,066 of them (7.2 % of all GT) belong to a track that is detected in
   the previous or next val_fast frame (4 frames apart); test clips are consecutive frames (spacing 1). A propagation
   step would need box tracking across frames, which is listed as out of scope (bonus) in `CLAUDE.md`. **Rejected.**
7. **Duplicates and NMS.** Duplicate FPs (a second box on an already matched GT) are 2,529 of 352,886 scored boxes
   (0.7 %); NMS is not leaving many same-class duplicates. OPT-02 (2-model ensemble) gave at best +0.0029.

## Main bottleneck

Precision, not recall or the cap: the pipeline scores many confident false positives, and the clearest, cheapest
removable group is the cut-off boxes on the inner tile borders (13 % of the scored boxes, almost no TPs).

## Ranked interventions

| # | intervention | expected val gain | risk | GPU time | test inference | defense complexity |
|---|---|---|---|---|---|---|
| 1 | **Drop (or down-weight) tile boxes touching an inner tile edge before the merge** | +0.003…+0.008 (removes 13 % of scored boxes, ≤ 162 TPs at stake, and most of those objects are also seen by another view) | low: CPU post-processing only, standard practice in sliced inference (SAHI-style) | 0 | unchanged (≈ 29 min) | very low |
| 2 | Source weighting / dropping single tiles | +0.001 (best single drop +0.0009) | medium: tuned per tile position on 714 frames, likely a side effect of #1 | 0 | ≤ unchanged | low |
| 3 | Short fine-tune of Model A at low LR | unknown, curve suggests ≤ 0 | high: overfitting seen after epoch 1 | 1–2.5 h | unchanged | medium |
| 4 | Temporal propagation across frames | ≤ 7 % recall ceiling, realistic gain unknown | high, out of scope (tracking) | 0 | + tracking | high |
| 5 | Top-500 allocation / class calibration | ≤ +0.0001 (oracle) | low | 0 | unchanged | low |
| 6 | RT-DETR-x | unknown | high: 8 GB GPU, > 4 h, no time | > 4 h | ≈ 2× slower | medium |

Selected: **#1**, frozen in `selection_rule.md` before any candidate was evaluated. Post-selection note: the
hypothesis was formed on Model A's val_fast tables, and the candidates are also screened on val_fast; val_full is the
held-out check.
