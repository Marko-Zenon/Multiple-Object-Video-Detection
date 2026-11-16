# DO NOT USE — CPU pipeline check, not a model result

8 training images, imgsz 320, 2 epochs, on the CPU while CH-07 owned the GPU. It only shows that
`scripts/train_ultra.py` and `scripts/predict_ultra.py curve` work end to end: data.yaml, label cache, `epoch<N>.pt`
checkpoints, a resumable curve and the evaluator integration. Both epochs hold the same weights, because Ultralytics
accumulates gradients to a nominal batch of 64 (`nbs`) and the 4 iterations per epoch gave one optimizer update.
