# OPT-10 chunk 10.4 + 10.5 — replicate R1 and candidate m2 (NEGATIVE: m2 rejected at S1)

Authorized by Badabuba on 2026-11-11, continuing from `4066c4c` on `opt-10-soup-bn`. Worker: Claude.
Rule: [`../selection_rule.md`](../selection_rule.md), unchanged. Contract: [`contract.json`](contract.json).
Decision record: [`decision.json`](decision.json). c0 stays rejected (S2); c0, D1, D2 and their evidence untouched.

## Outcome

`m2` (Model A + R1 soup) is **rejected at S1**: discovery mAP 0.325863 vs E6 0.334488, delta **-0.008625**
(threshold +0.0010). The delta is also below the barrier -0.0020, so the frozen rule rules out R2 / R3.
S2, S3, G1-G9, determinism, test inference and CSV were **not run**. No Kaggle CSV exists for this chunk.
D1 did not show H1 as material, so the rule's fallback applies: stop optimising, keep E6 + E4 as the selections.

## R1 (configs/runs/opt10_rep_s43.json)

| Item | Value |
|---|---|
| Recipe | Model A's `final_rtdetr.json`; the only Ultralytics kwarg that differs is `seed` 42 -> 43 (checked) |
| Start weights | `outputs/weights/rtdetr-l.pt` sha256 `6de60b10…e73f` (= Model A's init) |
| Data | train_final, 6,072 frames, export manifest equal to Model A's, split file sha256 `3a377223…a86a` |
| Schedule | 6-epoch schedule, stopped after epoch 2 (`stopped_after_epoch_2`); 12,144 images seen |
| Losses | finite; first 36.75, last 10.94, min 5.54, max 173.27 |
| Checkpoints | epoch 1 `epoch0.pt` `0e17db9e9c0c20dd146ece954d8c57b1fe69ca39f3bf5f3f0aa916d005d862c9`; epoch 2 `epoch1.pt` `5c02bd245b189f508403cbaf0cf5f3633635f9209d96ec778fcc15617ee4dce8` |
| Runtime / memory | 52.9 min training (RTX 5070 Laptop), peak 3.31 GB allocated / 3.46 GB reserved |
| Resume | a stop_after_epoch run is never resumed (train_ultra.py); resume was verified in the smoke |

Smoke (`replicates/smoke.json`): 100 frames x 2 epochs at seed 43, stop after epoch 1 then `--resume` (start epoch 1,
575 optimizer-state entries restored), finite losses, epoch0/epoch1 checkpoints, 8-frame inference finite, 3.98 GB peak.

### R1 alone (single full-frame view on val_fast, D1 eval-mode path; reported only, never a gate)

| Epoch | R1 val_fast mAP | Model A val_fast mAP | delta |
|---|---:|---:|---:|
| 1 | 0.249933 | 0.278967 | -0.029034 |
| 2 | 0.250523 | 0.258465 | -0.007942 |

R1's epoch 1 is much weaker than Model A's; Model A's epoch-1 checkpoint (weight 0.8 in E6) looks like a favourable
draw of the recipe rather than its typical result. That is consistent with m2's discovery loss.

## m2 build (`../m2/build.json`)

0.4 A_e1 (`6ff4a430…5b0b`) + 0.4 R1_e1 + 0.1 A_e2 (`5594fb0e…5799`) + 0.1 R1_e2, `movd.opt10.weighted_state`
(float32, stored float16). sha256 `7392eab92e50ffa9c2b857b9818d6f6b1b9af56fe5c6cec6ed4b24475bc91c1c`.
All 819 floating tensors averaged (max error vs an independent recomputation 0.0); the 122 integer
`num_batches_tracked` buffers equal Model A's; all finite; the four source checkpoints unchanged after the build.

## S1 — discovery (4 clips, 1,335 frames; E6 = unchanged E6 on the same frames)

| Metric | E6 | m2 | delta |
|---|---:|---:|---:|
| mAP | 0.334488 | 0.325863 | **-0.008625** |
| AP50 | 0.660794 | 0.647786 | -0.013008 |
| AP75 | 0.293135 | 0.275885 | -0.017249 |
| AP_small | 0.194868 | 0.199016 | +0.004147 |
| AP_medium | 0.335314 | 0.324726 | -0.010588 |
| AP_large | 0.544898 | 0.534216 | -0.010683 |

Per clip (reported only): 086 +0.021599, 339 +0.000215, 182 -0.002145, 137 -0.016077
(`../m2/discovery_per_clip.json`). Discovery raw cache: 1,335 frames, 614 s, 0.86 GB peak (`../m2/cache_discovery.json`).

Comparison with Model A: Model A has no discovery number of its own in OPT-10 (E6 is Model A's run inside the frozen
E6 pipeline). Model A's public 0.2774301362 and val_full 0.2860 are below E6 (val_full 0.2966, public 0.2947349001).

## GPU and checks

Phase 1 GPU 69.8 of 85 min, one process at a time, `nvidia-smi` before every job (`../replicates/nvidia_before_*.txt`):
smoke 2.7, R1 training 53.0, m2 discovery cache 10.2, R1 val_fast 3.9 min. Test-inference budget unused.
Start and end identity verify (`../identity.json`), tests, dataset, splits and scoped Ruff: [`checks/`](checks/).

## Code changes in this chunk

`scripts/opt10_soup.py`: `rep-smoke`, `train`, `build` (bn:false soups), `valfast`. One implementation defect fixed
before m2 was built: replicate checkpoint paths are stored relative to the hw2 folder and were joined to the repo
root (`run_checkpoint`); regression test added. No frozen value, threshold, frame list or protected file changed.
