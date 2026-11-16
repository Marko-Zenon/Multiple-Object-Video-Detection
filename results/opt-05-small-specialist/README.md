# OPT-05 — FB as a small-object specialist on top of Model A (negative result)

Optional post-plan experiment and the last optimization cycle, requested by Badabuba on 2026-11-10. Worker: Claude. CPU only.

**Outcome: V0. No candidate passes the val_fast gate.** The best gain is +0.0011 mAP (s5), and the gate needs +0.003.
Per the frozen rule there was no val_full run, no GPU use, no FB val_full or test inference, no CSV and no upload.
Model A (val_full 0.2860, public 0.2774301362, CSV `e587c68b…0da7`) stays the primary selection.

## What was tested

Rule, grid and gates: [`selection_rule.md`](selection_rule.md) and [`candidates.json`](candidates.json), committed in `04147c9`
before any merge was evaluated. Code: `movd/specialist.py`, `scripts/opt05_specialist.py` (committed in `a91bce7`, also before any result),
tests `tests/test_opt05.py`.

All Model A detections are kept unchanged. Only FB boxes that are *small* under a size rule are added, with FB's score times a
fixed multiplier. Then class-wise NMS and the global top-500 cap are applied. Boxes stay in original-pixel xywh.

* `coco32`: predicted box area < 32² px² in original pixels (the COCO small definition).
* `in32`: box < 32² px² at Model A's stretched 1344×1344 input, i.e. `w·h/(W·H) < (32/1344)²`.

## val_fast results (714 frames, project evaluator)

Gate V: mAP ≥ 0.2922166 · AP_S ≥ 0.1427339 · ≥100 bin ≥ 0.3465018 · width ≤ 1920 subset ≥ 0.3120912 · deterministic rerun.

| id | rule | m | NMS | mAP | Δ vs A | AP_S | AP_M | AP_L | ≥100 | subset ≤1920 | FB added | A removed | Gate V |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Model A | — | — | — | 0.28922 | — | 0.14273 | 0.25935 | 0.49013 | 0.35150 | 0.31209 | — | — | ref |
| FB alone | — | — | — | 0.27607 | −0.01314 | 0.15772 | 0.24517 | 0.45140 | 0.34692 | 0.29279 | — | — | (OPT-04) |
| s1 | coco32 | 1.00 | 0.5 | 0.28954 | +0.00032 | 0.15412 | 0.25987 | 0.49013 | 0.35132 | 0.31266 | 39,949 | 36,030 | fail (mAP) |
| s2 | coco32 | 1.00 | 0.6 | 0.28949 | +0.00027 | 0.15407 | 0.25980 | 0.49013 | 0.35130 | 0.31251 | 41,193 | 37,197 | fail (mAP) |
| s3 | coco32 | 0.75 | 0.5 | 0.28978 | +0.00056 | 0.14730 | 0.25949 | 0.49013 | 0.35149 | 0.31289 | 24,070 | 20,147 | fail (mAP) |
| s4 | coco32 | 0.75 | 0.6 | 0.28972 | +0.00050 | 0.14731 | 0.25935 | 0.49013 | 0.35148 | 0.31271 | 25,832 | 21,834 | fail (mAP) |
| s5 | in32 | 1.00 | 0.5 | 0.29029 | +0.00108 | 0.15046 | 0.26190 | 0.49013 | 0.35219 | 0.31114 | 46,635 | 43,585 | fail (mAP, subset) |
| s6 | in32 | 1.00 | 0.6 | 0.29026 | +0.00104 | 0.15035 | 0.26191 | 0.49013 | 0.35223 | 0.31101 | 48,540 | 45,425 | fail (mAP, subset) |
| s7 | in32 | 0.75 | 0.5 | 0.29027 | +0.00105 | 0.14420 | 0.26094 | 0.49013 | 0.35186 | 0.31206 | 27,246 | 24,218 | fail (mAP, subset) |
| s8 | in32 | 0.75 | 0.6 | 0.29027 | +0.00106 | 0.14419 | 0.26086 | 0.49013 | 0.35186 | 0.31189 | 29,587 | 26,478 | fail (mAP, subset) |

(Rounded to 5 decimals; exact values are in `screening_val_fast.csv` and `val_fast/<id>/metrics.json`. Gates use the unrounded values in `selection_val_fast.json`.)

Every candidate passes AP_S, the ≥100 bin and determinism. All fail the +0.003 mAP threshold. The `in32` candidates also fail
the width ≤ 1920 subset guard: their gain comes from the 3840-wide clip, which test does not have.
`passing = []`, `selected = null`.

## Observations (facts only; interpretation is for the students)

* FB's small boxes raise AP_S by up to +0.0114 (s1: 0.1541 vs 0.1427), but overall mAP rises by at most +0.0011.
* A is at the 500 cap in most frames, so every added FB box removes one A box (FB added ≈ A removed + boxes that NMS merged).
* AP_L moves by at most 0.00006 (0.49008–0.49014 vs 0.49013); large objects are almost untouched, as designed. Per-class changes are ≤ +0.010 (tricycle) and ≥ −0.003 (bicycle).

## Checks

* `identity.json`: both checkpoint sha256 and all 4 input prediction hashes match. Model A and FB val_fast reproduce their registered metrics exactly.
  Model A's width ≤ 1920 subset is 0.3120912393562929, the same as OPT-04's.
* A-only merge: unchanged at NMS 0.6. At NMS 0.5, 8 of 352,886 low-score A boxes (max score 0.079) are removed. Their stored
  pair IoU is about 0.49998; CH-13's NMS ran on the original xyxy, and the float32 xywh → xyxy rebuild lands just above 0.5.
  The mAP effect is +1e-9 (0.2892166451 vs 0.2892166441). This is recorded as a deviation; the frozen rule was not changed.
* `determinism.json`: an independent second process (`screen --rerun`) gives identical prediction sha256, metrics and subset values for all 8 candidates.
* Regenerate: `python scripts/opt05_specialist.py verify && python scripts/opt05_specialist.py screen && python scripts/opt05_specialist.py screen --rerun && python scripts/opt05_specialist.py determinism && python scripts/opt05_specialist.py select`
  (about 25 min on CPU; merged predictions go to the git-ignored `outputs/opt05/`).

## Known limitations

Cached predictions were already NMS-merged and capped at 500 inside each model's pipeline, so boxes that FB's own cap removed cannot come back.
The candidate was chosen on val_fast (714 frames, a subset of val_full). The size rule uses the predicted box, while AP_S uses the GT area.

## Final selections (rule default row, unchanged)

Primary **Model A** `e587c68b517d6d59f90122946d8a45ef5beac4fb8ae3b24f66ced1268def0da7` (val_full 0.2860, public 0.2774301362).
Secondary **F** `73f73e219dfbf3b674277703f7314d14c316d0800bcd046634aaecd27f2b793b` (val_full 0.2696, public 0.2328849249).
The students set them in the Kaggle web UI.

## TODO (students)

* TODO(student): interpret why higher AP_S from FB does not lift overall mAP (cap displacement? duplicates? the AP_S weight in mAP?).
* TODO(student): decide whether OPT-05 is mentioned in the report and how.
