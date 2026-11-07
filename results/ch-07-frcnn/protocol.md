# CH-07 Faster R-CNN experiment protocol

All three configurations use torchvision Faster R-CNN ResNet-50-FPN v2
with COCO_V1 initialization, a fresh eight-class head, CH-04 AMP and AdamW,
and CH-02's seed 42, horizontal flip, ignore-region treatment and frozen
1344-pixel long side. The batch size is two. Each full run sees 2,800 frozen
training frames for six epochs (16,800 images seen). The evaluator is CH-03
with the global top-500 cap.

| Run | Change relative to corrected baseline | Hypothesis |
|---|---|---|
| frcnn_baseline | CH-04 model defaults | Establish the corrected reference |
| frcnn_rpn_pre3000 | Training RPN pre-NMS candidates 2,000 → 3,000 | Preserve more candidate regions for tiny objects during training |
| frcnn_roi_pos50 | RoI positive sample fraction 0.25 → 0.50 | Sample more foreground RoIs when many annotated objects compete for the fixed RoI batch |

Marko's v0 RPN experiment used 3000/1000/2000/1000 and his v0 RoI
experiment used 256 samples at fraction 0.50. These runs isolate a single
parameter from each legacy idea. They keep the remaining proposal and RoI
settings at the baseline defaults. The v0 RPN result improved AP_small;
the v0 RoI result was negative. Those observations motivate the hypotheses.
The v0 run used 560 frames, three epochs, an 896-pixel long side, different
ignore handling and a 100-detection cap. See plan.md §2 for the historical
table. It is a separate protocol.

Select the final-epoch checkpoint with the highest final-epoch val_fast
mAP, with baseline first on an exact tie. All configurations get the
required val_full accuracy and CH-03 speed artifacts. Full validation and
speed are reported separately from selection. The score threshold used in
the qualitative figures applies to the figures only.

Before each full run, copy the corresponding configuration into ignored
outputs/ with its run ID suffixed _smoke, epochs set to 1, and
smoke_steps_per_epoch set to 50. Run the preflight with
python scripts/train_tv.py --config outputs/<run_id>_smoke.json --training-only.
Launch the full run with python scripts/train_tv.py --config configs/runs/<run_id>.json.
Use the same command with --resume when a completed epoch checkpoint exists.
The weights cache is set via TORCH_HOME and remains under ignored outputs/.
The SHA-256 of fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth is
dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf.

The preflight estimate was 3 × 119 minutes plus three short smokes,
based on CH-04, so the inherited 2.5-hour GPU estimate was optimistic.
