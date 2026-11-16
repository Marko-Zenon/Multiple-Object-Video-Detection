# CH-08 results

Selection uses final val_fast mAP at epoch 6. The anchor variant won
(0.17053) and reached 0.16925 mAP on val_full, +0.00693 vs the RetinaNet
baseline. AP50 improved from 0.31368 to 0.34649. AP75 slipped from
0.14580 to 0.14459 and AP_small from 0.07801 to 0.07541, while AP_large
increased from 0.30452 to 0.32226. The smaller anchor scales did not
improve small-object AP in this run. Bicycle AP increased from 0.1130
to 0.1535 and tricycle AP from 0.1463 to 0.1711. Van AP fell from 0.1913
to 0.1679.

Reducing gamma to 1.5 lowered overall mAP by 0.00991 vs baseline,
contrary to the primary hypothesis, while raising AP_small by 0.00518.
Car AP rose from 0.3476 to 0.3593, while truck AP fell from 0.1386 to
0.1038 and bus AP from 0.0524 to 0.0077. The bus class has 264 targets.

The selected Faster R-CNN reached 0.18934 mAP and 0.10829 AP_small,
ahead of the selected RetinaNet by 0.02008 and 0.03288 respectively.
It also led in all eight class APs. RetinaNet anchors led on AP_large
(0.32226 vs 0.31066) and measured 19.71 vs 14.85 FPS. On the 189
frames with at least 100 targets, the anchor variant reached 0.26028
mAP, versus 0.22810 baseline, 0.21655 focal and 0.27852 selected Faster
R-CNN. Density bins contain different image and class mixes.

See comparison.csv, per_class.csv, density.csv, and the val_fast figure.
FPS is batch-one on the same RTX 5070 with 30 GPU-resident frames,
including model preprocessing and NMS. Single-run timings can vary with
system conditions and detector postprocessing load. Each experiment
used one seed and the final checkpoint. No causal claim about speed
or statistical significance follows from these measurements.

Full-run wall times including validation and speed were 3996.66,
3464.18 and 2641.30 seconds, about 168.4 minutes combined.
Each 50-step smoke used two 25-step epochs, forcibly stopped after
epoch 1 and resumed with the original configuration plus --resume.
All reached step 50 with finite loss, wrote usable checkpoints, and
passed original-pixel prediction and 500-detection cap checks on
four val_fast frames. Each full run completed all 8,400 steps and
16,800 images without interruption.
