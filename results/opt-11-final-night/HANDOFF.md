# OPT-11 hand-off (2026-11-14 23:03:19 Kyiv)

Nothing was uploaded. Kaggle final selections stay E6 + E4 until a new public score beats them.

## Protected CSVs (re-hashed now)

- E6_csv: `C:/UCU/AI/hw2/MOVD-opt08/submissions/opt08_e6_combination_k1_test.csv` sha256 `35a225c905ac56ed86532314046821d10085f73781666ee843fb4c6d28127c35` unchanged
- E4_csv: `C:/UCU/AI/hw2/MOVD-opt08/submissions/opt08_e4_checkpoint_soup_s1_test.csv` sha256 `7dc9b318e354d4839b5dffae8f5973dd7da234fdbeb3c5f84057694878852014` unchanged

## Validated new CSVs

| Candidate | CSV | sha256 | rows | detections | validators | val evidence |
|---|---|---|---|---|---|---|
| c1 | `C:/UCU/AI/hw2/MOVD-opt11/submissions/opt11_c1_w075_test.csv` | `04fcd221137f033af8557c561d6046d2fa3a08fcad8eb566380dfb6f91361e8c` | 3811 | 1846709 | project 0, independent 0 | val_full 0.297399 (+0.0008 vs E6), confirmation +0.0037 (OPT-09) |
| p2 | `C:/UCU/AI/hw2/MOVD-opt11/submissions/opt11_p2_trajsoup_test.csv` | `69fd9896ee645de5c7debae4f134bd0f092f35e02888a43479f83bdc8a5a7e2b` | 3811 | 1853738 | project 0, independent 0 | discovery delta +0.002468, 2/4 clips >= E6; val_full 0.297845 (E6 0.296583), confirmation 0.256107 (E6 0.270895) |
| c2 | `C:/UCU/AI/hw2/MOVD-opt11/submissions/opt11_c2_classmix_test.csv` | `3e25acda0794df5da3b097f43f9a567806109e82f6639cf34075c49c21fc14ab` | 3811 | 1831222 | project 0, independent 0 | val_full 0.297638 (optimistic: classes chosen on val) |

P decision: `{"screens": {"p1": {"delta": 0.001874215341411689, "clips_non_negative": 4}, "p2": {"delta": 0.002468267748375219, "clips_non_negative": 2}}, "rule": "best of p1/p2; promote iff delta >= 0.0011 and >= 2/4 clips >= E6; strong iff delta >= 0.0025", "decided_at_kyiv": "2026-11-14 22:13:17", "best": "p2", "promoted": true, "strong": false}`

## Recommended manual upload order

1. c1: `C:/UCU/AI/hw2/MOVD-opt11/submissions/opt11_c1_w075_test.csv`
2. c2: `C:/UCU/AI/hw2/MOVD-opt11/submissions/opt11_c2_classmix_test.csv`
3. p2: `C:/UCU/AI/hw2/MOVD-opt11/submissions/opt11_p2_trajsoup_test.csv`

Do not append to results/submissions.csv until the public scores are known.

## p2 flag (added 23:08 Kyiv, `valfull/p2_subsets.json`)

p2 vs E6 on val: val_full +0.0013, discovery +0.0025, confirmation -0.0148, width<=1920 -0.0002, without-4K +0.0022.
Per clip: 086 -0.0066, 117 -0.0175, 137 -0.0007, 182 +0.0008, 268 (4K) +0.0022, 305 +0.0105, 339 +0.0032.
It is unstable across clips and flat on the test-like width<=1920 subset, so it is ranked last.
p1 (discovery +0.0019, 4/4 clips non-negative) was not run on test: the frozen rule picked p2 by discovery delta,
and at 23:08 a 28-min test inference no longer fits the 23:30 deadline.

## Public Kaggle scores (user-reported, 2026-11-14)

| Candidate | val_full | Public | vs E6 0.2947349001 |
|---|---:|---:|---:|
| c1 | 0.297399 | **0.2960489045** | +0.0013 |
| c2 | 0.297638 | 0.2959983429 | +0.0013 (tied with c1, -0.00005) |
| p2 | 0.297845 | 0.2934521776 | -0.0013 |

Final Kaggle selections (student decision): **c1 + c2**, replacing E6 + E4.
c2 is effectively tied with c1 but slightly lower. p2 failed to transfer despite the highest val_full, in line with
its confirmation (-0.0148) and width<=1920 (-0.0002) flags. No further experiment.
