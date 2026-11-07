# CH-05 — test inference, submission writer and validator

Code: `movd/submission.py`, `scripts/{predict_test,make_submission,validate_submission,submission_roundtrip}.py`,
tests in `tests/test_submission.py`. The submission CSVs live in the git-ignored `submissions/` folder, and raw
predictions in `outputs/ch-05-submission/`. Neither is ever committed.

## Evidence in this folder

| File | What it is |
|---|---|
| `roundtrip_synthetic_seed42.json` | Val round-trip on `val_full` with seeded synthetic predictions (CPU, no model): writer → CSV → validator → parser → `movd.evaluate`. Max \|difference\| vs direct evaluation = 0 (tolerance 1e-6) |
| `DO_NOT_UPLOAD_synthetic_test/submission.json` | Full 3,811-row synthetic test submission. It exercises clipping, zero-area rejection, empty rows, the 500 cap and all 6 test resolutions. Validator OK. **Format test only, never upload** |
| `DO_NOT_UPLOAD_cpu_smoke_frcnn/{predict,submission}.json` | CPU smoke run of `predict_test.py` with the CH-04 **smoke** checkpoint (`smoke_frcnn_integrated`, 2×25 steps, random init) on 4 frames (clip_001 1920×1080, clip_011 680×382), then `make_submission.py`. Pipeline test only, **an untrained model, never upload** |

## Real submission (after a validated trained checkpoint exists and the GPU is free)

```bash
python scripts/predict_test.py --checkpoint outputs/<run_id>/epoch_<NNNN>.pt --name <run_id>_e<N>
#   resumable: rerun with --resume after a stop (finished clips are kept)
python scripts/make_submission.py outputs/ch-05-submission/<run_id>_e<N>/test_predictions.json
python scripts/validate_submission.py submissions/<run_id>_e<N>.csv           # must print [OK]
```

Before the first upload: check the daily limit on the Kaggle submit page (gate G2b) and do not guess it.
Upload through the Kaggle web UI only (no API token in the repository). Then append one row to
`results/submissions.csv` with the CSV sha256 printed by the validator, the checkpoint and its sha256 (from
`results/ch-05-submission/<name>/predict.json`), the checkpoint's `val_full` mAP from its run's `metrics.json`,
and the Kaggle public score and status.

## Format decisions

- Class = `category_id − 1` (0–7). Box = top-left `x y width height` in original pixels. Torchvision's model
  transform resizes and pads internally and maps boxes back to each image's original size; this is tested for
  all 6 test resolutions.
- Per detection: reject a non-finite score or one outside [0, 1]; reject a non-finite box or a negative
  width/height; clip to the image; reject zero area after rounding. Sort by full-precision score (ties: class,
  then box), keep the top 500 over all classes, and write an empty string when nothing is left.
- Precision: score 8 decimals, box 5 decimals. On the synthetic val set, 2, 3 and 4 box decimals changed mAP by
  2e-4, 3.4e-6 and 1.4e-5, because rounding flips borderline IoU matches. 5 decimals gave 0 difference, at about
  52 characters per detection.
