# OPT-08 e4_checkpoint_soup: E4 — weight soup of Model A with the adjacent epoch (3 frozen weights)

Written and committed **before** any soup checkpoint was built or run. Not changed after results are seen.

## Hypothesis

Model A is CH-14 epoch 1 (val_fast 0.279); epoch 2 (`epoch1.pt`) fell to 0.2585 while the warmup LR was still rising.
Both lie on one training trajectory, so a linear interpolation biased towards Model A may average out part of the
epoch-1 noise and keep some of epoch 2's extra fitting (AP_small 0.1079 -> 0.1089) without its loss.

## Compatibility (checked before anything else; `compatibility.json`)

Model A `epoch0.pt` sha256 `6ff4a430…5b0b` and `epoch1.pt` sha256 `5594fb0e…5799` (both = `results/ch-14-final/
checkpoints.json`): same class RTDETRDetectionModel, same YAML, the same 941 tensor names, shapes and dtypes
(float16 + int64), adjacent epochs (0, 1); weights in the `ema` entry, `model` is None in both. Result: compatible.
Had any check failed, E4 would have been rejected without inference.

## Exact deterministic rule (`scripts/opt08_soup.py`)

soup(w) = w x Model A + (1 - w) x epoch1.pt for every floating tensor (float32 arithmetic, stored as float16);
integer tensors from Model A. Saved as a checkpoint with `ema` = soup, no optimizer (`soup_build.json`: path and
sha256). Inference: the unchanged CH-13 pipeline (`configs/runs/final_inference.json`: full, flip, 4 tiles at imgsz
1344, decode top 300 / conf 0.001, class-wise NMS 0.5, top 500) through `movd.infer.raw_pass` per view, merged by
`opt03.edge_filter_combine` with factor 1 (= the unchanged CH-13 merge, OPT-07 identity); then x2 (IoU 0.5).

Frozen parameters (`movd/opt08.py` EXPERIMENTS):

    s1 {"iou": 0.5, "kind": "soup", "tag": "w080"}
    s2 {"iou": 0.5, "kind": "soup", "tag": "w065"}
    s3 {"iou": 0.5, "kind": "soup", "tag": "w050"}

## Order, smoke and cost

1. CUDA smoke: Model A through the same cache + merge path on 20 frames must equal Model A's stored predictions
   exactly (`smoke.json`), else E4 stops.
2. S1: raw caches of the 4 discovery clips (1,335 frames, ~11 GPU min per weight) for all 3 soups; selection and
   S1 rule as in TASK.md (x2 on the same discovery clips is the baseline).
3. S2: the selection's raw cache on the 3 confirmation clips (~12.5 GPU min), once.
4. S3: val_full (both caches, no new inference) once, U1-U6 / U8 vs x2.
5. U7 for E4: the CPU transform reproduced in a fresh process (val_full sha256) **and** a GPU re-inference of
   confirmation clip `uav0000305_00000_v` (184 frames) equal to the cached raw shard exactly (max abs diff 0).
6. Test raw cache (3,811 frames, ~31 GPU min) only if eligible; then the CSV and both validators.

Comparison baseline: OPT-07 x2 (val_full `6c3a929f…2c03`, public 0.2805168313). Every stage starts only if it can
finish with >= 30 min to spare before 07:20 Kyiv (TASK.md time contract).
