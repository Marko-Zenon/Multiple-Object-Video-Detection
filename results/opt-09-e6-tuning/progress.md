# OPT-09 progress log

Focused local search around the protected E6 pipeline. Worker: Claude for Badabuba. Branch `opt-09-e6-tuning`,
worktree `C:/UCU/AI/hw2/MOVD-opt09`. Rules: [`selection_rule.md`](selection_rule.md). Times are Europe/Kyiv.

## 2026-11-11 10:30 — claim and start checks

* `origin/develop` = a848486 (PR #26 merged 2026-11-11 04:28 UTC); OPT-08 ✅ DONE; no open PR into develop;
  notebook lock FREE; RTX 5070 idle (1.9 GB used by the desktop, 0 %), no Python process.
* Claim pushed to develop (9516b2f). The row first said 08:30: Git Bash ignored `TZ=Europe/Kyiv` and printed UTC.
  The claim was at 10:30 Kyiv (commit time 10:29:54 +02:00). Budget: until 13:30 Kyiv.
* `check_dataset.py` PASS; `make_splits.py --check` PASS (val_fast / val_full hashes as committed).
* Kaggle public scores (user-reported) added to `results/submissions.csv`: E6 0.2947349001, E4 0.2913788894,
  x2 0.2805168313, x4 0.2787106940. E6 and E4 are the 2 current Kaggle final selections (user-reported).

## 10:40 — identity (`identity.json`)

All protected hashes match: Model A `6ff4a430…5b0b`, epoch1.pt `5594fb0e…5799`, soup w080 `2d60d3fa…d939`,
E6 val_full `bdb2806e…ad52`, E6 test `ef3f5c2f…0aa`, E6 CSV `35a225c9…7c35`, E4 CSV `7dc9b318…2014`,
x2 CSV `9dc8fb03…c867`, x4 CSV `09a5c8d1…2fcb`; 15 E6 raw shards hashed; frozen files equal `origin/develop`.
The OPT-09 code path (parallel merge + `opt09.five_frame_boost` at IoU 0.5, strength 1) rebuilds E6 val_full
byte-identically (367 s CPU).

## Next

Commit `selection_rule.md` + this log + code before any evaluation; then baseline, Phase B build + smoke, Phase A.
