# CH-09 export and adapter evidence

All of this was produced on the CPU (`CUDA_VISIBLE_DEVICES=-1`) while CH-07 owned the GPU.

| File | What it shows | Command |
|---|---|---|
| `export_check.json` | Masked YOLO export of `train_rubric` (2,800 frames) and the unmodified `val_fast` copy (714) checked against the COCO JSON: per-class label counts equal the non-crowd targets, class ids 0–7, label boxes within 0.003 px, a pixel check on 40 masked frames, and disk use | `python scripts/export_yolo.py --dataset-root <dataset>` |
| `figures/export_check.png` | Visual policy check on the 2 frames with the most fill: ignore regions (red) are filled, labelled targets (green) are kept | same |
| `adapter_check.json` | (1) GT → YOLO labels → Ultralytics `Results` → `movd.ultra.result_to_coco` → 500 cap → `movd.evaluate` on `val_full`: mAP 1.0. (2) CPU-only predict with COCO-pretrained `yolo11s.pt` on 3 original val frames: boxes in original pixels, category ids 1–8, accepted by the evaluator. This checks the format only, not accuracy | `python scripts/check_ultra_adapter.py --dataset-root <dataset> --model-preflight outputs/weights/yolo11s.pt` |

The exported data itself is in the git-ignored `outputs/yolo_data/` (about 1.9 GB) and is never committed.
