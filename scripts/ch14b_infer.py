"""CH-14b inference: the unchanged CH-14 runner (`scripts/ch14_infer.py`) with CH-14b output folders.

    python scripts/ch14b_infer.py val  --weights <ckpt> --name stageA_epoch1 --split val_fast --device 0
    python scripts/ch14b_infer.py test --weights <ckpt> --name <name> --device 0

Same commands and arguments as `scripts/ch14_infer.py` (CH-13 pipeline `configs/runs/final_inference.json`,
project evaluator, resumable per-clip shards). Only two folders change, so the CH-14 evidence is never touched:
evidence -> `results/ch-14b-rtdetr-short-finetune/<name>/<split>/`, shards -> `outputs/ch14b/<name>/<split>/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ch14_infer  # noqa: E402

from movd import data  # noqa: E402

ch14_infer.RESULTS = data.REPO_ROOT / "results" / "ch-14b-rtdetr-short-finetune"
ch14_infer.OUTPUTS = data.REPO_ROOT / "outputs" / "ch14b"

if __name__ == "__main__":
    ch14_infer.main()
