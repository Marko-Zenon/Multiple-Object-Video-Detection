# OPT-09 selection rule — focused local search around E6 (frozen before any evaluation)

Written and committed **before** any OPT-09 candidate was evaluated on any split, and before any new soup was built
or run. Not changed after a result is seen. Requested by Badabuba (Nazar) on 2026-11-11. Worker: Claude.
Code: `movd/opt09.py` (grids, boost), `scripts/opt09_loop.py` (stages, gate), `scripts/opt09_soup.py` (Phase B).

## Protected baseline: E6 (OPT-08 `e6_combination` k1), never overwritten

| Item | Value |
|---|---|
| Pipeline | soup w080 -> CH-13 merge (`configs/runs/final_inference.json`: full + flip + 4 tiles at 1344, class-wise NMS 0.5, top 500) with inner tile-border boxes dropped (`opt03.edge_filter_combine`, factor 0, margin 0.005 of the tile side) -> OPT-08 b3 five-frame persistence boost |
| Boost (b3) | box supported if its best same-class match (first maximum) in t-1 **and** t+1 has IoU >= 0.5; mean = own + t-1 + t+1 scores, plus t-2 / t+2 match scores where those also have IoU >= 0.5; score := max(own, mean). No explicit multiplier exists: the implicit strength is 1 (the score is raised all the way to the mean) |
| Soup w080 | 0.8 x Model A `epoch0.pt` (CH-14 epoch 1, `6ff4a430…5b0b`) + 0.2 x `epoch1.pt` (CH-14 epoch 2, `5594fb0e…5799`); `MOVD-opt08/outputs/opt08/soup/soup_w080.pt` sha256 `2d60d3fa66d9fd37032bcaaaeb297e98510217b600b5809f73873b36360cd939` |
| val_full predictions | `MOVD-opt08/outputs/opt08/e6_combination/k1/val_full.json` sha256 `bdb2806edad09fa093d5b59369780a8dbbb6fea4758522c72c86b8c8e98cad52`; val_full mAP 0.29658283872855484 |
| test predictions | `MOVD-opt08/outputs/opt08/e6_combination/k1/test/test_predictions.json` sha256 `ef3f5c2ff42598696dfcd44aa1431c0d422598203bbfce8f0370c333c14cb0aa` |
| CSV | `MOVD-opt08/submissions/opt08_e6_combination_k1_test.csv` sha256 `35a225c905ac56ed86532314046821d10085f73781666ee843fb4c6d28127c35`, 3,811 rows, 1,841,598 detections; Kaggle public **0.2947349001** (user-reported) |
| Raw caches | `MOVD-opt08/outputs/opt08/soup/raw/w080/{discovery,confirmation,test}/*.npz` (per-shard sha256 in `identity.json`) |

`identity.json` (checked before this rule): every hash above matches, the frozen files equal `origin/develop`, and
the OPT-09 code path rebuilds the E6 val_full predictions byte-identically (sha256 `bdb2806e…ad52`).

## Values already tested (never repeated)

* Border margin with drop (factor 0): 0.002, 0.005, 0.01, 0.02 (OPT-03 e1-e4, Model A, val_fast); OPT-07 x4 and
  OPT-08 E1 / E6 used 0.005 only. Down-weighting (factor 0.5 / 0.25) was OPT-03 e5 / e6; OPT-09 keeps factor 0.
* Persistence boost: matching IoU 0.5 only (OPT-07 x2, OPT-08 E3 / E6); strength 1 only. Support rules b1-b4 (OPT-08 E3).
* Soup weight on Model A: 1.0 (Model A), 0.8, 0.65, 0.5 (OPT-08 E4: discovery +0.0030, +0.0024, +0.0001 vs x2).

## Frozen grids (each candidate changes exactly one E6 parameter; `movd.opt09.GRIDS`)

    a1_border_margin/m1 {"margin": 0.0035}
    a1_border_margin/m2 {"margin": 0.0065}
    a1_border_margin/m3 {"margin": 0.008}
    a2_persistence/p1 {"strength": 0.75}
    a2_persistence/p2 {"strength": 1.25}
    a2_persistence/p3 {"strength": 1.5}
    a2_persistence/p4 {"iou": 0.45}
    a2_persistence/p5 {"iou": 0.55}
    b_soup_weight/w1 {"soup": "w075"}
    b_soup_weight/w2 {"soup": "w085"}
    b_soup_weight/w3 {"soup": "w090"}

* **A1 border** (3 candidates): only the margin; border boxes are still dropped; soup w080 and b3 unchanged.
* **A2 persistence** (5 candidates, two sub-grids, never both at once): strength λ in {0.75, 1.25, 1.5}, where a
  supported box with mean > own gets score := min(1, own + λ (mean - own)) (λ = 1 is E6 exactly, tested in
  `tests/test_opt09.py`); or the matching IoU in {0.45, 0.55} for every neighbour match (t-1, t+1, t-2, t+2).
  No box is generated, moved or removed. Soup w080 and border 0.005 unchanged.
* **B soup weight** (3 candidates): w in {0.75, 0.85, 0.9} on Model A, the nearest untested weights around 0.8
  (OPT-08 discovery peaked at 0.8 with 0.65 close behind; 1.0 is Model A). Same OPT-08 arithmetic
  (`opt08_soup.soup_state`, float32, stored float16), border 0.005 and b3 unchanged. No training in OPT-09.

## Stages (the OPT-08 split, `results/opt-08-overnight/split.json`, reused unchanged)

Discovery = clips 339, 086, 137, 182 (1,335 frames); confirmation = 305, 117, 268 (4K) (1,511 frames).
Baseline for every stage = unchanged E6 on the same frames (`baseline/subsets.json`, must equal the OPT-08 E6 record).

* **S1 (discovery):** every candidate of a grid on the discovery clips only. Selection = highest discovery mAP
  (ties: the earlier candidate). S1 passes if the selection's delta vs E6 >= +0.0003.
* **S2 (confirmation):** the selection only, once, on the confirmation clips. Passes if its delta vs E6 is > 0.
* A component **passes** only if S1 and S2 pass. A grid that fails stops; it is never retuned or re-screened.
* For Phase B, each soup is built and smoked before inference, and each soup's discovery raw cache is the only
  GPU work before S1; the confirmation raw cache is built only for the S1 selection, and test only for an eligible
  Phase C candidate. val_full is evaluated only in Phase C (after S1 and S2 passed).

Phase B smoke (`opt09_soup.py build` + `smoke`, before any discovery cache): parameter names, shapes and dtypes equal
Model A's; rebuilding w080 with the same arithmetic gives the E6 soup's tensors exactly; 20 GPU frames of w080 equal
OPT-08's raw shard exactly; 20 frames of each new soup are finite with scores in [0, 1]. Any failure: Phase B stops.

## Phase C (fixed now; at most 2 final candidates; no tuning after a result)

* **c1** = E6 with the passed component whose S1 discovery delta is largest (ties: the earlier phase A1, A2, B).
* **c2** = E6 with all passed components together, only if at least two passed (order soup -> border -> boost).
* No component passed: NEGATIVE result, no Phase C, no CSV.

## Promotion gate vs unchanged E6 (val_full + subsets, once per Phase C candidate)

| Check | Rule |
|---|---|
| G1 | val_full mAP >= E6 + 0.0010 |
| G2 | width <= 1920 subset mAP >= E6 + 0.0005 |
| G3 | at least 5 of 7 clips non-negative |
| G4 | the gain does not depend on the 4K clip: val_full without clip 268 (3840 px) >= E6 + 0.0010 |
| G5 | AP_small loss <= 0.002 |
| G6 | density >= 100 objects bin loss <= 0.002 |
| G7 | all 8 class APs reported (`per_class.csv`) |
| G8 | a separate-process rerun gives byte-identical val_full predictions and the same test predictions as the CSV run |
| G9 | evaluator, protocol, ground truth, ignore policy, class map and CSV code unchanged (`git diff origin/develop` on `movd/evaluate.py`, `movd/submission.py`, `movd/data.py`, `movd/infer.py`, `configs/`, both validators) |
| also | discovery and confirmation deltas both > 0 |

A candidate is eligible only if every check passes. Eligible: lossless test predictions and CSV, the project
validator and the independent validator; path, filename, sha256, checkpoint composition, rule, row and detection
counts recorded; **never uploaded**. At most 2 new CSVs. No candidate eligible: stop with a negative result; the
gate is not weakened and no further experiment is added.

## Time and GPU

Total OPT-09 budget 3 h from the claim (10:30 Kyiv) -> work ends by 13:30 Kyiv. One GPU process at a time,
`nvidia-smi` checked before each; a GPU job starts only if its expected time (discovery ~10 min per soup,
confirmation ~12.5 min, test ~29 min, from OPT-08) fits before 13:15. Phase B discovery caches may be generated
while Phase A runs on the CPU (inference only, no Phase B result is looked at before Phase A is decided).
