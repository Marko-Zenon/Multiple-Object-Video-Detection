# OPT-12 hand-off (2026-11-15 10:25:06 Kyiv)

Nothing was uploaded. c1 (public 0.2960489045) and c2 (0.2959983429) stay the selected, upload-ready fallbacks; their CSVs are unchanged (see protected_*.json).

Training result: `done`. Reference R (Ultralytics monitor of w075) = 0.2814225018426861.

| Epoch | monitor mAP50-95 | epoch s | BN stats frozen |
|---|---|---|---|
| 1 | 0.2701 | 1581.6 | True |
| 2 | 0.2752 | 1503.7 | True |
| 3 | 0.2837 | 1479.4 | True |
| 4 | 0.2800 | 1455.3 | True |
| 5 | 0.2712 | 1456.8 | True |
| 6 | 0.2731 | 1454.7 | True |
| 7 | 0.2747 | 1456.5 | True |
| 8 | 0.2586 | 1459.4 | True |
| 9 | 0.2647 | 1438.6 | True |
| 10 | 0.2602 | 1440.9 | True |

Selection (single view, val_fast, project evaluator): w075 0.2882; curve e1 0.2759, e2 0.2809, e3 0.2916, e4 0.2865, e5 0.2778, e6 0.2797, e7 0.2814, e8 0.2656, e9 0.2718, e10 0.2671; AVG3 0.2725. N-star = best (0.2916, screen pass). alpha-star = 0.5 (W 0.3045, screen pass).

## Candidates (deltas vs c1 on val; gates frozen in selection_rule.md)

| Candidate | val_full | w<=1920 | w/o 4K | discovery | confirmation | clips >= 0 | gate | CSV |
|---|---|---|---|---|---|---|---|---|
| C1 (nstar) | +0.0066 | -0.0003 | +0.0119 | +0.0034 | -0.0351 | 2/7 | fail | - |
| C2 (wstar) | +0.0148 | +0.0137 | +0.0189 | +0.0133 | -0.0267 | 5/7 | fail (confirmation); student exception 2026-11-15 10:21:22 | ready |
| C3 (ens) | +0.0090 | +0.0029 | +0.0143 | +0.0049 | -0.0308 | 4/7 | fail | - |
| C4 (w070) | +0.0000 | +0.0000 | +0.0001 | -0.0003 | +0.0003 | 3/7 | fail | - |

## Validated CSVs

- **C2**: `C:/UCU/AI/hw2/MOVD-opt12/submissions/opt12_c2_wiseft_test.csv`  
  sha256 `9e2ed403e495bf507005a0675d423ab82788e18a28185734011eced152a95fb4`, 1812330 detections, 3811 rows; validators project 0 / independent 0; rebuild identical: True

Advisory low-resolution check (width<=1920 val clips down-scaled 2x, delta vs c1; demote below -0.005, never a gate): C1 +0.0253, C2 +0.0476, C3 +0.0284

## Recommended manual upload order

1. C2

C1 was not built (gate failed), so the frozen rule's branch 'C2 first' applies.
Keep c1 selected until a new public score beats 0.2960489045; the two final selections remain the students' decision.
