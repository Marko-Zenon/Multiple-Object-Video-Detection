# CH-04 GPU smoke and resume evidence

Run from the repository root with the CUDA environment active.

- `python scripts/check_dataset.py` verified the extracted dataset.
- `python scripts/make_splits.py --check` verified all three frozen split hashes.
- `python -m pytest tests/test_tv_train.py -q --basetemp outputs/ch04-pytest-fixed` passed all three tests.
- `python -u scripts/train_tv.py --training-only --config configs/runs/smoke_frcnn.json`
  ran in a child process. After the checkpoint message for epoch 1 at global step 25,
  the controller called `Popen.kill()` and observed exit code 1 on Windows.
  The same command with `--resume` printed `Resuming at epoch 2, global_step=25`
  and completed at step 50. Both checkpoints remained readable.
- `python -u scripts/train_tv.py --training-only --config configs/runs/smoke_retina.json`
  completed both epochs and all 50 steps without interruption.

Both runs used two spawned DataLoader workers, batch size 2, AMP, random
weights and the frozen resize, train list and ignore treatment. These short
runs measure throughput and checkpoint behavior. Their losses do not measure
detector quality. CUDA RoI Align backward warned about nondeterminism despite
fixed seeds and deterministic settings where supported.

| Family | Epoch 1 mean loss | Epoch 2 mean loss | Train wall time, 100 images | Images/s | Peak allocated GPU GiB | Projected 16,800-image training time |
|---|---:|---:|---:|---:|---:|---:|
| Faster R-CNN v2 | 1.98089 | 1.93074 | 38.25 s | 2.61 | 4.29 | 107 min |
| RetinaNet v2 | 1.52264 | 1.26599 | 33.40 s | 2.99 | 4.20 | 94 min |

Training times include loading, masking and worker startup and exclude checkpoint
writes and evaluation. Extrapolation assumes constant throughput. Worker startup
has a higher per-image cost here because each epoch is only 25 steps. The smoke
runs verify that the frozen input size and batch fit on the RTX 5070 Laptop GPU.
The one-hour target remains unconfirmed. Keep the CH-02 image budget and resize
until an authorized protocol decision, and budget for validation separately.
Training-only JSON artifacts mark evaluator and speed results as pending.
