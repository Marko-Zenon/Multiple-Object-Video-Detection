# CH-12 finalist-selection rule (written 2026-11-09 13:20 Kyiv, before the aggregation was built)

plan.md does not contain a separate numeric finalist rule for CH-12. This rule uses only items plan.md already freezes, and it is applied
as written. Where a source is given, the criterion comes from there.

1. **Comparable evidence only** (CLAUDE.md §8, plan.md §4).
   - A run enters the primary comparison only if every compatibility check in `compatibility.csv` passes: val_full, all 2,846 frames,
     0 predictions outside the split, global maxDets 500 with 0 images over the cap, the provided COCO GT with crowd copies (CH-02
     ignore policy), `train_rubric` x 6 epochs = 16,800 images seen, long side / imgsz 1344, seed 42, and a recorded checkpoint, model
     configuration and hardware.
   - A run that fails a check is still listed, with the failed check named. Examples: `yolo_imgsz1600` (resolution exception),
     `rtdetr_lr1e4` (confounded) and smoke runs.
2. **Primary metric: val_full mAP@[.5:.95]** from `movd/evaluate.py` (plan.md §4 "Primary: mAP", "Every reported headline number uses
   val_full"). Each family is represented by the run its own chunk selected (CH-07 frcnn_rpn_pre3000, CH-08 retina_anchors, CH-10 best at
   1344 = yolo_p2, CH-11 rtdetr_baseline). All other runs are shown too.
3. **Not one metric alone.**
   - The leader on mAP must also lead on most of the 9 secondary checks: AP50, AP75, AP_S, AP_M, AP_L and the 4 density bins.
     Every check it loses is reported.
   - Its mAP lead must exceed the measured run-to-run spread: 0.0063 val_full mAP, CH-10 replicate.
   - A lead inside that spread, or lost on most secondary checks, counts as a tie. A tie is decided by throughput on the same GPU,
     and the decision note says it is a tie.
4. **Feasibility** (plan.md §6-7 and the CH-12 acceptance criterion "runtime estimate that fits the schedule").
   - The `final.json` training config must fit inside CH-14's window with a buffer for one resume, estimated from measured throughput
     of the chosen family on the RTX 5070. Final training starts no later than Mon 22:00, no new training jobs start after Tue 08:00,
     and experiments stop Tue 12:00.
   - Any change from the validated configuration (data amount, epochs, input size) must be justified by a cited result. Otherwise the
     validated value is kept.
5. **Resolution exception.** The CH-10 `yolo_imgsz1600` result is reported separately as resolution sensitivity. It can move the
   final input size only if (a) its family is the finalist, or the finalist has its own evidence at that size, and (b) rule 4 permits
   the extra compute.
6. **Second final selection** (G2c: 2 selections, by internal validation). `final.json` names the CH-14 fallback checkpoint
   (plan.md §7 "Final training late: submit the best CH-10 / CH-11 checkpoint"): the rubric-budget checkpoint of the finalist.
