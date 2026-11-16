# rtdetr_lr1e4: kept, marked CONFOUNDED (not a clean one-variable comparison)

This full run is complete and valid as a measurement (`metrics.json`, `curve.json` and `speed.json` come from the project
evaluator and benchmark), but it changed **two** training variables against `rtdetr_baseline`, not one:

| variable | rtdetr_baseline (`optimizer=auto`) | rtdetr_lr1e4 |
|---|---|---|
| peak lr0 | 0.000833 (AdamW, chosen by auto) | 0.0001 (AdamW, explicit) |
| bias warmup start (`warmup_bias_lr`) | 0.0 (auto sets it) | 0.1 (Ultralytics default, kept) |

Cause: in Ultralytics 8.4.207, `BaseTrainer.build_optimizer` sets `self.args.warmup_bias_lr = 0.0` only inside the
`optimizer == "auto"` branch. Naming the optimizer explicitly (needed so that `lr0` is not ignored) skips that branch.
The run config's note ("warmup ... stays the same") was written before training and is wrong on this point; the config is left
unchanged because it is the record of what ran. The effect is visible in `ultralytics/results.csv`: the bias group
(`lr/pg2`) is at 0.0667 after epoch 1, against 3.3e-05 for the weights and 0.00028 for the baseline's bias group.

The clean comparison is `results/rtdetr_lr1e4_wb0/` (`configs/runs/rtdetr_lr1e4_wb0.json`): the same settings plus
`warmup_bias_lr 0.0`, so that only lr0 differs from what `optimizer=auto` picks (checked by
`tests/test_rtdetr.py::test_optimizer_auto_matches_the_explicit_learning_rate_variant_except_lr0`).
