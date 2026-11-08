# CH-07 failure cases

The selected checkpoint is epoch 6 of frcnn_rpn_pre3000. The cases below
use original-pixel coordinates. Green marks the inspected ground truth,
white marks other annotated targets, red marks the shown prediction, and
shaded gray marks an ignored region. The scan uses predictions with score
at least 0.10 and checks same-class IoU at 0.50. This visual diagnostic uses
its own display threshold. Reported AP comes from the project evaluator.

1. [Tiny person localization](figures/tiny_target_miss.png). The 15 × 36 px
   target is surrounded by overlapping people. The 0.80-score person box
   shifts toward a neighbor and reaches only 0.38 IoU. Limited pixels after
   resize and the overlapping people are plausible localization causes.
2. [Car/van confusion](figures/car_van_confusion.png). The box overlaps the
   annotated car at 0.78 IoU, but the detector assigns van at 0.63 score.
   The tall boxy rear, nighttime lighting and aerial angle plausibly make
   the class cues ambiguous. This example has good spatial overlap.
3. [Localization in a dense frame](figures/dense_frame_miss.png). The frame
   contains 93 targets and seven ignore regions. A 0.97-score person box
   covers parts of two adjacent pedestrians and overlaps the inspected
   target at 0.43 IoU. Their overlap and a dark background plausibly
   explain the oversized box despite high classification confidence.

The figures show plausible causes in individual frames. Their contribution
to aggregate AP was not measured separately.
