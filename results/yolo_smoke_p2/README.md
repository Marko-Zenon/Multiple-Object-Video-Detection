# yolo_smoke_p2: FAILED compatibility smoke (kept as evidence, do not use)

Kept on purpose. The plumbing part of the smoke passed: build, Ultralytics `Transferred 563/563`, training, resume after
epoch 0 with 255 optimizer entries, prediction and the project evaluator. But its starting checkpoint was wrong, so the
P2 variant was **not** launched from it.

**What went wrong.** The first `movd/yolo_p2.py` built the plain 8-class yolo11s with *placeholder* class names and then
called Ultralytics' `load`. The baseline trainer sets the dataset names first
(`DetectionTrainer.set_model_names_for_load`), and `load` then copies the COCO classification rows of the classes
that share a name (person, bicycle, car, truck, bus). With placeholder names `load` skips that step, but the log still
reads `Transferred 499/499`. As a result every class row of this init was untrained, not only the 3 classes without a
COCO name.

**How it was found.** After 50 images this smoke scored 0.0103 val_fast mAP, against 0.1223 for the 1600 smoke. In
both smokes the optimizer had effectively not stepped yet: batch 1 with `nbs` 64 accumulation, the EMA updated once
or twice, Adam step count 0. So the gap had to come from the starting weights. A forward-pass check showed the P3-P5
Detect inputs and outputs were bit-identical to a rebuild of the baseline, and switching the P2 level off changed nothing
(0.0104). The P2 branch carries only a small share of the gradient at initialisation (0.16-6.5 of a total norm of
74-220 on 3 real frames), so gradient clipping was ruled out. Comparing the parameters with the trainer-built baseline
start (`outputs/yolo_smoke_1600/.../epoch0.pt`, no optimizer step taken) showed that all 8 class rows differed.

**Fix.** `build_p2_init` sets the names before `load`, and it checks that the matched rows equal the COCO rows
(`cls_rows_copied_from_coco_by_name` in `results/ch-10-yolo-tuning/p2_transfer.json`). The fixed init now differs from
the trainer's baseline start only in rows 3, 5 and 7 (van, tricycle, motor). Those rows are randomly initialised in both,
because they have no COCO counterpart. The re-run smoke is `results/yolo_smoke_p2_fixed/`.

Files: `p2_transfer_placeholder_names.json` is the transfer report of the bad init (sha256 `be0d8ae7…` in
`config.json`). The bad init is kept, git-ignored, as `outputs/weights/DO_NOT_USE_yolo11s-p2-init_placeholder_names.pt`.
