"""Run one torchvision experiment from a JSON configuration."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-id", help="Use a new run directory with the same settings"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Smoke test without evaluator or speed metrics",
    )
    args = parser.parse_args()
    from movd.tv_train import train

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.run_id is not None:
        config["run_id"] = args.run_id
    train(config, resume=args.resume, training_only=args.training_only)


if __name__ == "__main__":
    main()
