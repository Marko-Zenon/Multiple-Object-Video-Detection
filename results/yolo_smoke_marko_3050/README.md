# Marko RTX 3050 smoke evidence — not baseline results

Authorized handoff pushed to develop: `7e3b440`, 2026-11-08 21:21 Kyiv. CPU implementation `8e48881`
is preserved. GPU execution belongs to Marko-Zenon; CH-08 remains Badabuba's separate GPU job.

Dataset check passed: train 24,201 / 56 clips, val 2,846 / 7, test 3,811 / 48; annotations and paths accessible.
Export passed: 2,800 train images / 137,756 targets, 714 val_fast images / 28,685 targets, classes 0–7,
zero duplicate labels, <=0.00288 px coordinate error. Masked 2,161 train frames; 40-frame pixel checks
and two-frame visual comparison passed. Original CPU export evidence remains intact in ch-09-yolo.
New export evidence is in `results/ch-09-yolo/marko_export/`; actual export stays ignored in `outputs/yolo_data/`.

Official COCO-pretrained YOLO11s SHA-256/source/size are in weights.json. Training initialized an eight-class YOLO11s head; the pinned trainer logged
499/499 transferred parameter items both initially and on resume.
See original console logs for actual transfer counts. No alternate training model was substituted.

| Setting | Actual value |
|---|---|
| Config | configs/runs/yolo_smoke_marko_3050.json |
| Resolution / batch | 1344 / 1 |
| AMP / workers / image cache | true / 0 / false |
| Seed / augmentation | 42 / frozen flip-only |
| Subset | fraction 0.018, 50 first sorted frames from train_rubric |
| Initial / resumed training | 50 + 50 batches, 100 total; same 50 images twice |
| Planned epochs / graceful interruption | 2 / after epoch 0 checkpoint save |
| Native resume | start_epoch 1; 255 restored optimizer state entries |
| Initial finite total batch loss range | 4.321056–6.448348 |
| Resumed finite total batch loss range | 4.053787–6.448348 |
| Initial mean box / cls / dfl loss | 2.08054 / 1.90381 / 1.33526 |
| Resumed mean box / cls / dfl loss | 2.08242 / 1.90033 / 1.32979 |
| Initial train epoch time / throughput | 6.65 s / about 7.52 batches/s, includes warmup |
| CLI process times | 17.978 s initial + 77.083 s resumed (includes internal validation) |
| Peak CUDA allocated / reserved | 1,376,612,352 / 1,574,961,152 bytes (1.28 / 1.47 GiB) |
| Peak total device use, sampled | 1,799 MiB; includes context and other GPU users |
| Peak process tree RSS | 2,997,063,680 bytes (2.79 GiB) |
| Minimum system available RAM | 1,929,363,456 bytes (1.80 GiB) |

All 100 post-backward batch losses were finite. Both monitored commands exited 0, no OOM/guard stop.
Runtime JSON records callback observations; resource summaries distinguish CUDA allocator peaks from
one-second device/RSS sampling. Monitoring cannot guarantee capture of sub-second RSS/device peaks.
Nearly all 4 GiB swap was already used before training and remained so: RAM remains a risk.

An unfinished epoch-0 last.pt was copied to `outputs/yolo_smoke_marko_3050/train/weights/resume_source_epoch0.pt`
before resuming. It is 94,823,715 bytes and contains epoch 0, 255 optimizer entries and scaler state
(scale 32768); its hash is in resume_source.json. Native resume continued epoch 1 and wrote epoch1.pt.
This verifies epoch-boundary optimizer checkpoint recovery, not a crash in the middle of a batch.
Completed last.pt/best.pt have optimizer state stripped by Ultralytics; use the preserved original
copy to inspect unfinished state. All five checkpoint files remain ignored and inventoried by SHA-256.

Ultralytics performed final internal validation on 714 images despite val=False, then validated best.pt.
The numbers in the copied CSV/console are not project headline metrics. No project val_fast curve,
val_full evaluation, speed benchmark or failure analysis has been claimed for this smoke.
No full baseline training, notebook edits, completion PR, or external artifact transfer occurred.

Validation: CH-09 unit/integration tests 39 passed; complete CPU suite 149 passed / 2 skipped
(CUDA parity hidden deliberately; optional ungenerated sample submission). Ruff check with RUF100
ignored for the existing repository noqa convention and format check passed. Dataset check passed.
Exactly 13 missing packages were added; no existing package version changed, including torch 2.14.1+cu130
and torchvision 0.29.1+cu130. pip check retains the same unrelated inherited environment errors listed
in pip-check-before.txt; it reports no new dependency conflict. CUDA and CUDA NMS passed (prerequisites.json).

Next: review this smoke and choose recorded laptop batch/worker/cache settings for the full six-epoch,
2,800-image baseline while retaining resolution 1344, seed 42 and the frozen augmentation/budget.
The shared baseline still uses batch 4/workers 2 and was not changed or run. Fifty frames from one clip
do not establish worst-case dense-frame memory safety or full-baseline throughput. Recheck free RAM/GPU,
stop and retain diagnostics on OOM. CH-09 remains incomplete pending baseline/curve/evaluation and integration.

Transfer: full console logs, resource samples and helper are under outputs/ch09_marko_setup_20261108/;
checkpoints are under outputs/yolo_smoke_marko_3050/train/weights/. The prepared smoke_artifacts manifests
list relative paths, byte sizes and SHA-256. Once Badabuba supplies an authorized SSH/private-cloud
destination, copy the listed files into a fresh directory and verify every hash there. No upload started.
