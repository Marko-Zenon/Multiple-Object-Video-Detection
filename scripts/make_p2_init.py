"""Build the CH-10 YOLO11s-P2 starting checkpoint (CPU) and record the partial weight transfer.

    python scripts/make_p2_init.py

Writes the git-ignored `outputs/weights/yolo11s-p2-init.pt` and the tracked report
`results/ch-10-yolo-tuning/p2_transfer.json`. Deterministic for a fixed seed (protocol seed 42).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import data, yolo_p2  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=data.REPO_ROOT / "outputs" / "weights" / "yolo11s.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=data.REPO_ROOT / "outputs" / "weights" / "yolo11s-p2-init.pt",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=data.REPO_ROOT / "results" / "ch-10-yolo-tuning" / "p2_transfer.json",
    )
    args = parser.parse_args()
    report = yolo_p2.build_p2_init(args.pretrained, args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"transferred {report['transferred_tensors']}/{report['p2_tensors']} tensors "
        f"({report['transferred_elements']:,} of {report['p2_state_elements']:,} state-dict elements); "
        f"fresh P2 branch: {report['fresh_tensors']} tensors, {report['fresh_elements']:,} elements; "
        f"strides {report['detect_strides']} -> {args.out}"
    )


if __name__ == "__main__":
    main()
