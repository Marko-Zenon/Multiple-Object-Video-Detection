# OPT-10 Phase 0 evidence checkpoint

Continuation of Fable checkpoints e588d8e / cffa98e by Codex for Badabuba on 2026-11-11, in the existing opt-10-soup-bn branch and MOVD-opt10 worktree.

## Outcome

c0 passes frozen S1 discovery: 0.335996060383 vs E6 0.334488045029, delta +0.001508015353 >= +0.0010. H2 is material under the frozen reading, and m4bn is eligible under its discovery rule. This grants no approval for a later chunk.

Confirmation admission was refused by the unchanged GPU budget guard: 41.235 minutes used + 13-minute confirmation reservation > 45-minute Phase 0 cap. Actual Phase 0 use is 2474.1 seconds. Confirmation inference and S2 were not run. S3 and the promotion gate were not run. c0 is not upload-ready and has no val_full result. Chunk 10.3 is partial, awaiting student review of the budget stop.

## Identity and smoke

- Start identity passed. Every protected OPT-09 artifact and all 55 protected w080 raw shards match. E6 full-validation predictions rebuilt byte-identically; sha256 bdb2806edad09fa093d5b59369780a8dbbb6fea4758522c72c86b8c8e98cad52.
- Generalized 0.8 / 0.2 checkpoint average equals all 941 protected w080 tensors exactly.
- Raw-cache smoke passed on 20 frames / six views, exactly equal to the OPT-08 protected shard.
- Unfused Model A validation anchor differs from its stored fused mAP by +0.000023267300, within tolerance 0.0005.

## D1 and D2

D1 uses 714 val_fast frames, with normal stored BN statistics and per-image BN statistics through the same unfused path. D2 uses the frozen 714 unmasked train_final original frames in normal eval mode. All rows use full-frame 1344, top 300, confidence 0.001 and the unchanged project evaluator.

| diagnostic | epoch | mode | mAP | AP50 | AP75 | AP_small | AP_medium | AP_large |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| D1 | 1 | stored BN | 0.278989974 | 0.516095718 | 0.257890563 | 0.107883554 | 0.246310240 | 0.483463131 |
| D1 | 1 | per-image BN | 0.256616625 | 0.483606889 | 0.237922234 | 0.107412336 | 0.218843204 | 0.462744827 |
| D1 | 3 | stored BN | 0.261349820 | 0.495466966 | 0.236716648 | 0.104738742 | 0.229135709 | 0.453362464 |
| D1 | 3 | per-image BN | 0.243914599 | 0.460449446 | 0.221715700 | 0.102733557 | 0.208397629 | 0.450250126 |
| D1 | 6 | stored BN | 0.238084752 | 0.451995900 | 0.215634658 | 0.089985754 | 0.211189755 | 0.411337339 |
| D1 | 6 | per-image BN | 0.233885170 | 0.444121546 | 0.216109834 | 0.096400930 | 0.197767723 | 0.426711342 |
| D2 | 1 | stored BN, train frames | 0.397074777 | 0.692873962 | 0.399839886 | 0.249959559 | 0.390724972 | 0.580053381 |
| D2 | 3 | stored BN, train frames | 0.477392397 | 0.801686188 | 0.494474800 | 0.323713383 | 0.483948795 | 0.636769752 |
| D2 | 6 | stored BN, train frames | 0.499405821 | 0.825606311 | 0.523386757 | 0.338506376 | 0.505452235 | 0.654685267 |

| epoch | stored fused val_fast mAP | D1 unfused minus stored | per-image minus eval mAP | state restored byte-identically |
|---|---:|---:|---:|---|
| 1 | 0.278966706958 | +0.000023267300 | -0.022373348868 | True |
| 3 | 0.261371115575 | -0.000021295945 | -0.017435220565 | True |
| 6 | 0.238126383260 | -0.000041630980 | -0.004199582088 | True |

Frozen diagnostic readings: H1 not supported (epoch-6 lift -0.004199582088 < +0.010). H3 confirmed by the frozen criterion (train-frame mAP rise +0.102331043934 >= +0.03, alongside declining validation mAP). H2 material at S1 (discovery delta >= +0.0010). The diagnostic readings are reported as defined in the frozen rule; no diagnostic was used to select a checkpoint.

## c0 build and discovery metrics

c0 checkpoint sha256: 6acb20696921a1b5e1fd7b63d0d17d76f6349d7a1f1eba785f0d0f441a16002c. Exactly 1,024 frozen masked train_final frames, 122 BN layers, cumulative statistics, FP32, batch one, no gradients, original weights retained. All 122 counters equal 1,024; modes and momenta restored; all non-BN tensors identical to w080. Image-content digest: b8c9bb9d89a7b5266608df63dc2b1fa750d0f19c40f8482c327adc0a5b1bf69f.

| metric | E6 discovery | c0 discovery | delta |
|---|---:|---:|---:|
| mAP | 0.334488045029 | 0.335996060383 | +0.001508015353 |
| AP50 | 0.660794031231 | 0.662996452603 | +0.002202421372 |
| AP75 | 0.293134625115 | 0.294752072140 | +0.001617447025 |
| AP_small | 0.194868293365 | 0.191686739537 | -0.003181553828 |
| AP_medium | 0.335314255845 | 0.336941446880 | +0.001627191035 |
| AP_large | 0.544898386541 | 0.540036665726 | -0.004861720815 |

S1 checks mAP only. The discovery AP_small delta of -0.003181553828 is reported without treating it as a val_full G5 result. Confirmation and val_full remain unmeasured.

## Implementation corrections

1. 47e0f39 corrected a nonexistent standalone CH-14 manifest reference. The actual masked export manifest is compared to the committed manifest object inside CH-14 export_check.json. The frozen frame list and input images did not change. The failed attempt stopped before the CUDA context.
2. ebb72d8 corrected a strict-positive variance validity check. The same eight constant BN channels have zero variance in c0 and protected w080 (seven in model.6.ec.bn, one in model.7.ec.bn), with positive epsilon 0.001 and no negative variances. Zero variance is valid. The completed c0 checkpoint was CPU-audited and accepted unchanged, without a BN rerun. The original build_FAILED.json and variance_audit.json are retained. No experiment candidate, threshold or model tensor was changed by this correction.

## Validation and handoff

- Full existing suite on CPU: 345 passed, 4 skipped. Skips cover CUDA-only / missing reference-weight fixtures; detailed reasons in checks/pytest.txt. Phase 0 itself exercised CUDA through smoke, D1, D2, BN and discovery inference.
- Full dataset check and make_splits.py --check passed.
- Scoped Ruff check (ignoring RUF100 as in OPT-09) and format --check passed on the four OPT-10 Python files.
- Start and end identity verifications both passed (identity.json). Final rebuild 356.6 seconds; protected E6 predictions are byte-identical and all protected hashes match.
- No replicate training, chunk 10.4, confirmation inference, test inference, submission or upload was performed. Protected E6, E4, Model A, epoch1.pt and w080 artifacts matched again at the end identity check.

Technical evidence only. Student conclusions and decisions remain with Nazar and Marko.

GPU reservation released on origin/develop at 15:06 Kyiv in dbef9fd. Task remains PARTIAL / PAUSED under Badabuba for student review.
