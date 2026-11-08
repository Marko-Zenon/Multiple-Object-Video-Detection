"""Regenerate a checkpoint's val_full predictions and check them against its run's metrics.json (CH-05).

    python scripts/predict_val.py --checkpoint outputs/<run_id>/epoch_0006.pt --name <run_id>_e6

This uses the same code path as training-time validation in `movd.tv_train` (`validation_frames`,
`make_loader`, `predict_coco`), so the predictions are those the CH-07 evaluation scored. They feed
`scripts/submission_roundtrip.py --predictions ...` before the first real test submission.

Output (git-ignored): `outputs/ch-05-submission/<name>/val_full_predictions.json`.
Evidence (tracked): `results/ch-05-submission/<name>/val_check.json`: direct `movd.evaluate` metrics,
the metrics stored in `results/<run_id>/metrics.json` and their absolute difference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from predict_test import load_model, sha256  # noqa: E402

from movd.evaluate import evaluate_split  # noqa: E402
from movd.tv_train import (  # noqa: E402
    make_loader,
    predict_coco,
    save_json,
    validation_frames,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (movd.tv_train.predict_coco is CUDA-only)")

    start = time.perf_counter()
    checkpoint_sha = sha256(args.checkpoint)
    model, config, info = load_model(args.checkpoint, torch.device("cuda"), False)
    if args.dataset_root is not None:
        config = {**config, "dataset_root": str(args.dataset_root)}
    frames = validation_frames(config, "val_full")
    predictions = predict_coco(
        model, make_loader(frames, config), config["family"], config["amp"]
    )
    out = Path("outputs") / "ch-05-submission" / args.name / "val_full_predictions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, predictions)

    result = evaluate_split(predictions, split="val_full", root=args.dataset_root)
    stored = json.loads(
        (Path("results") / config["run_id"] / "metrics.json").read_text(
            encoding="utf-8"
        )
    )["val_full"]
    diff = {
        k: abs(result["metrics"][k] - stored["metrics"][k]) for k in result["metrics"]
    }
    evidence = Path("results") / "ch-05-submission" / args.name
    evidence.mkdir(parents=True, exist_ok=True)
    save_json(
        evidence / "val_check.json",
        {
            "checkpoint": args.checkpoint.as_posix(),
            "checkpoint_sha256": checkpoint_sha,
            "model": info,
            "predictions_file": out.as_posix(),
            "num_images": result["num_images"],
            "num_predictions": len(predictions),
            "stored_num_predictions": stored["num_predictions"],
            "direct": result["metrics"],
            "stored_ch07": stored["metrics"],
            "abs_difference": diff,
            "max_abs_difference": max(diff.values()),
            "seconds": time.perf_counter() - start,
        },
    )
    for k in result["metrics"]:
        print(
            f"{k:<8} direct {result['metrics'][k]:.6f}  CH-07 {stored['metrics'][k]:.6f}"
        )
    print(
        f"{len(predictions)} predictions ({stored['num_predictions']} in CH-07); "
        f"max |difference| = {max(diff.values()):.3g}; wrote {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
