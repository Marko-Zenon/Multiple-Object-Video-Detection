# Legacy configs (not used)

`data.json` and `experiments.json` were in `configs/` before CH-02. They do **not** describe the v0 run in
`homework.ipynb`, which hard-coded its settings in cells. Examples of the mismatch:

| Setting | `experiments.json` | v0 notebook |
|---|---|---|
| Train frames per clip | 4 | 10 |
| Optimizer | SGD, lr 0.002, momentum 0.9 | AdamW, lr 1e-4 |
| Faster R-CNN loss experiment | GIoU regression | RPN smooth-L1 β = 1/3 |
| RetinaNet focal γ experiment | 1.0 | 1.5 |

`data.json` also gives a 0.2 validation fraction and an 800–1280 px input, neither of which the notebook used.
No code reads these files. They are kept only for history.

The frozen protocol is `configs/protocol.json`, and the fixed frame lists are in `configs/splits/` (both from CH-02).
