# CH-09 transfer preparation — no transfer performed

The original baseline inventory (`results/yolo_baseline_marko_3050/artifact_manifest.json`, 145 files,
1,339,757,445 bytes) is preserved and reverified. The supplemental `transfer_manifest.json` adds the
executed notebook, closeout verification/tests, source/configuration and the original manifests.
`transfer_manifest.sha256` and `transfer_files.txt` use repository-relative paths.

Required payload includes all six full-baseline epoch checkpoints, last.pt/best.pt, their optimizer/scaler
state where retained, six raw val_fast prediction files, raw val_full predictions, full training/evaluation
logs and resource samples, configurations, package/hardware metadata, project metrics, per-class/density
results and all figures. Preserved laptop smoke outputs and its actual unfinished recovery checkpoint
are also included. Completed last.pt/best.pt are optimizer-stripped by Ultralytics; each epoch checkpoint
retains optimizer/scaler state. The eight full-run checkpoints were loaded and inspected on CPU again.
No dataset, YOLO export, package cache or credentials are included. No external destination is assumed.

## Command template after destination agreement and authorization

Run from the repository root, after confirming an SSH host/key and a fresh destination directory with Badabuba.
Replace HOST and the path with the agreed values. These commands have **not** been executed:

```bash
CH09_DEST='badabuba@HOST:/absolute/path/to/fresh-ch09-directory/'
rsync -a --partial --protect-args \
  --files-from=results/ch-09-yolo/notebook_closeout/transfer_files.txt \
  ./ "$CH09_DEST"
rsync -a --protect-args \
  results/ch-09-yolo/notebook_closeout/transfer_manifest.json \
  results/ch-09-yolo/notebook_closeout/transfer_manifest.sha256 \
  results/ch-09-yolo/notebook_closeout/transfer_files.txt \
  "$CH09_DEST"
```

Badabuba then checks the received payload from that fresh directory:

```bash
sha256sum -c transfer_manifest.sha256
```

Require every entry to pass, compare the inventory file count/byte total, and record Badabuba's verification
before claiming transfer completion. Transfer and recipient verification remain **PENDING**. Keep local
originals and failed/partial transfers until successful receipt is confirmed. A mutually agreed private
cloud destination is an alternative, with the same integrity checks; no public checkpoint links are needed.

## Review state

The five CH-09 technical acceptance checks and the evidence-only notebook integration are verified.
The existing 90 notebook cells/outputs and metadata are unchanged; three appended cells execute only
stored-evidence display, with five embedded figures. No training or inference was run in this session.
The full CPU test suite passed (149 passed, 2 skipped); existing GPU benchmark evidence remains in the
baseline results. Scoped CH-09 lint/format passes. Broader pre-existing repository lint findings are
preserved and documented in results/yolo_baseline_marko_3050/lint_summary.json.

A draft review PR may be opened into develop, but CH-09 must remain incomplete until student acceptance,
verified checkpoint receipt and the repository merge/completion procedure are satisfied. CH-10/11 remain
blocked; no new task is started. Student interpretation/conclusions are left as notebook TODOs.
