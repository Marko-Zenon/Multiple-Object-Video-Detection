# CH-08 RetinaNet preparation

The three configurations reuse the CH-04 model builder and trainer. Each
uses COCO_V1 RetinaNet ResNet-50-FPN v2, seed 42, 16,800 training images
seen, batch 2, AMP, learning rate 0.0001, resize 756–1344, CH-02 masking
and the CH-03 evaluator with the global 500-detection cap. The baseline
uses default anchors and focal alpha 0.25, gamma 2.0. Both variants keep
the same input, optimizer and schedule.

## Anchor decision

Run `python results/ch-08-retina/derive_anchors.py` from the worktree.
The script clips valid non-crowd boxes using CH-02's target helper, maps
them to the frozen model resize, and summarizes square-root area and
height-to-width ratio. The median side is 32.75 px on 137,756 usable
targets from 2,800 frames. To target that scale with the middle P3 anchor,
divide by the standard 2^(1/3) multiplier, then round to the nearest
multiple of four, yielding a base of 24 px. At each of five pyramid
levels, double the base and keep the standard three scales per octave.
The resulting groups are [24,30,38], [48,60,76], [96,120,152],
[192,241,304], [384,483,609].

The ratio quartiles are 0.69, 1.10 and 1.79; 69.0% fall between 0.5
and 2.0, and all three defaults have substantial nearest-ratio support.
Keep 0.5, 1 and 2 at every level to isolate the scale change. Nine
anchors per location preserve classification and regression tensor
shapes. Shape statistics do not establish recall or AP improvements.
Regression tensors are reused with changed anchor geometry, so the
initial decoded boxes can differ.

## Focal-loss hypothesis

Change only gamma from 2.0 to 1.5, keeping alpha at 0.25. Lower gamma
reduces the attenuation of easier examples. The hypothesis is that
this may improve overall AP when moderately easy foreground still
provides useful gradients. Additional background gradient could instead
hurt precision, particularly for small objects. Compare with the
corrected baseline on AP and AP_small.

Marko's v0 notebook supplied the smaller-octave-anchor idea and the
gamma 1.5 focal experiment. Its reported v0 baseline / anchors / focal
mAP was 0.1075 / 0.1025 / 0.1163. The shorter training, 896-pixel maximum,
ignore handling and 100-detection cap differ from this protocol.
CH-07's selected Faster R-CNN RPN variant provides the equal-budget
reference after CH-08 runs.

## Pretraining and next commands

CH-04 loads standard COCO weights before replacing the classification
predictor with eight foreground channels per anchor. The feature tower
and box regression head retain the pretrained weights. Gamma is a config
value and adds no state-dict entries. Resume with the original JSON so
the gamma and anchor geometry are restored as well as the tensors.

After confirming exclusive GPU access and before each full run, copy its
JSON into ignored outputs/, suffix the run ID with _smoke, set epochs=1
and smoke_steps_per_epoch=50. Run
`python scripts/train_tv.py --config outputs/<run_id>_smoke.json --training-only`.
Check finite loss, GPU memory and usable checkpoints. Then run sequentially:

```powershell
$env:TORCH_HOME = (Resolve-Path outputs/torch).Path
python scripts/train_tv.py --config configs/runs/retina_baseline.json
python scripts/train_tv.py --config configs/runs/retina_anchors.json
python scripts/train_tv.py --config configs/runs/retina_focal_gamma15.json
```

Add --resume to the same command if an epoch checkpoint exists. Reserve
up to about 116 minutes per run (5 h 48 min total before smokes), based
on CH-04's short-run projection. Startup overhead makes this estimate
conservative and the actual RetinaNet full-epoch pace is unmeasured.
These commands produce val_fast curves, final val_full metrics and FPS.
The comparison and notebook section follow the full runs.

## CPU verification

All three configurations loaded the official COCO_V1 weights on CPU.
Every state tensor outside the replaced classification logits matched the
original checkpoint exactly, and every adapted state dict reloaded
strictly into the same configuration. Weight SHA-256:
5905b1c544219215e544dbe319720397bc4e68de61a733a59350d7976645b769.
The focal unit test used identical state tensors and verified both gamma
values against torchvision's focal loss, including an ignored match.
CH-04 already supports both variants, so no trainer change was needed.
This turn used CPU only. CUDA smokes, memory checks, full training,
comparison and the notebook section remain pending.
