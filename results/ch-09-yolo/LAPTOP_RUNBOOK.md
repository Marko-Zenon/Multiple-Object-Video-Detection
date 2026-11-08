# CH-09 Ubuntu RTX 3050 smoke and recovery

This supplements GPU_RUNBOOK.md; its shared Windows/RTX 5070 commands and baseline configs are preserved.
The handoff was pushed to develop in `7e3b440` before installation or training. Marko-Zenon owns GPU execution;
Badabuba retains CH-08 on his separate machine. Full baseline training is pending review.

Use the existing `.venv` from the repository root. Ultralytics remains pinned to 8.4.207.
Set `PYTHONDONTWRITEBYTECODE=1`, and place `YOLO_CONFIG_DIR`, `MPLCONFIGDIR`, and `TORCH_HOME` under an ignored
`outputs/` setup directory. No dataset extraction or system/PyTorch reinstall is needed.

```bash
.venv/bin/python scripts/check_dataset.py
.venv/bin/python scripts/export_yolo.py --results-dir results/ch-09-yolo/marko_export
# Already executed; these paths now contain evidence. Never overwrite them to rerun.
.venv/bin/python scripts/train_ultra.py --config configs/runs/yolo_smoke_marko_3050.json --device 0 --stop-after-epoch 1
.venv/bin/python scripts/train_ultra.py --config configs/runs/yolo_smoke_marko_3050.json --device 0 --resume
```

The laptop config uses official COCO-pretrained YOLO11s, 1344 pixels, batch 1, AMP, workers 0, cache false,
seed 42, and frozen flip-only augmentation. Fraction 0.018 selects 50 of 2,800 training images.
Two planned epochs permit a genuine unfinished-checkpoint resume: 50 batches initially, then 50 resumed;
100 batches total. These are batches, not optimizer updates (nominal accumulation batch remains 64).
The subset is the first 50 sorted frames, so the test does not cover all scenes/densities.

`--stop-after-epoch 1` is allowed only for smoke configs with more than one planned epoch. It stops through
`on_model_save` after epoch 0 has been written, before finalization strips optimizer state. The preserved
`resume_source_epoch0.pt` is a byte copy of that `last.pt`. Resume uses the existing CLI's `--resume` and
native Ultralytics optimizer/scaler restoration. Runtime evidence records the actual starting epoch,
optimizer state count, finite batch losses, AMP, sample/batch count, and CUDA allocation peaks.
This is a graceful epoch-boundary interruption test, not a power-loss or mid-batch recovery guarantee.

A local process monitor sampled RSS, available RAM, swap, and total GPU memory every second. Its 768 MiB
available-RAM guard targets only its own child process group. Full logs and samples are in
`outputs/ch09_marko_setup_20261108/`; small summaries are copied into the smoke results directory.
Ultralytics may validate at the final epoch despite `val=False`, and validates stripped best.pt at exit.
Those internal metrics are plumbing evidence only; project headline metrics require movd.evaluate.

Do not launch the shared `yolo_baseline.json` unchanged on this laptop: it still has batch 4/workers 2.
After review, select a new recorded laptop baseline config with batch 1/workers 0/cache false/AMP and the
unchanged 1344 resolution, six epochs and full 2,800-image train_rubric. Recheck GPU/RAM first. Larger and
denser frames can exceed this smoke's memory use; stop and retain evidence on OOM. Resolution or budget
changes require student approval. Do not start baseline automatically.

## Artifact transfer (prepared only)

Keep the entire smoke output directory, original checkpoint copy, full setup logs/resource samples,
small results, run config, and pretrained source/hash record. `smoke_artifacts.sha256` lists relative paths;
`smoke_artifacts.json` also records file sizes. No weights, exports, caches or predictions belong in Git.
No raw predictions were produced by this smoke. Final baseline predictions and all final checkpoints/logs
must be added to a new manifest when available.

Once Badabuba supplies and authorizes an SSH destination, use rsync over SSH with a relative-path file list
made from the JSON manifest. Transfer into a fresh directory, preserve all existing runs, then run
`sha256sum -c smoke_artifacts.sha256` there. Verify file counts/sizes and every hash, including checkpoint
and original epoch-0 copy, before Badabuba loads anything. An agreed private cloud folder with the same
manifest verification is an alternative. Do not send credentials in logs or use public weight links.
No external transfer was initiated, and no destination is assumed.
