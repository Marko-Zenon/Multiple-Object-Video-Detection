"""Make the machine paths in finished RT-DETR run evidence repo-relative (CH-11 provenance hygiene).

    python results/ch-11-rtdetr/relativize_paths.py rtdetr_baseline [rtdetr_q500 ...]

Ultralytics records absolute `data`, `project`, `model` and `save_dir` paths in `results/<run>/config.json`
and `results/<run>/ultralytics/args.yaml`. This rewrites the repo-root prefix (the worktree that ran the job) to
`.`, so the evidence carries no user or machine path. Only path strings change: no metric, setting or hash.
Run it only after a run's training, curve and final steps are done (no script reads these fields back).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ("config.json", "ultralytics/args.yaml")


def prefixes() -> list[str]:
    native = str(ROOT)
    # args.yaml holds plain Windows paths, config.json holds them JSON-escaped; check POSIX spellings too.
    return [native.replace("\\", "\\\\"), native, ROOT.as_posix()]


def relativize(text: str) -> str:
    for prefix in prefixes():
        text = text.replace(prefix, ".")
    return text


def main(runs: list[str]) -> int:
    if not runs:
        print(__doc__)
        return 2
    for run in runs:
        for name in FILES:
            path = ROOT / "results" / run / name
            raw = path.read_bytes().decode("utf-8")
            new = relativize(raw)
            if new != raw:
                path.write_bytes(new.encode("utf-8"))
            left = [p for p in prefixes() if p in new]
            print(
                f"{run}/{name}: {'relativized' if new != raw else 'unchanged'}; absolute prefixes left: {len(left)}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
