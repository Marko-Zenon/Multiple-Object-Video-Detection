"""Build CH-09 numeric comparison evidence from stored project-evaluator results.

No model selection or student-authored conclusions are generated.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = (
    "frcnn_baseline",
    "frcnn_rpn_pre3000",
    "frcnn_roi_pos50",
    "yolo_baseline_marko_3050",
)


def main():
    rows = []
    for name in RUNS:
        folder = ROOT / "results" / name
        stored = json.loads((folder / "metrics.json").read_text())
        metrics = stored.get("val_full", stored)
        speed = json.loads((folder / "speed.json").read_text())
        timing = speed.get("inference", speed)
        assert metrics["split"] == "val_full" and metrics["num_images"] == 2846
        assert metrics["max_dets"] == 500
        rows.append(
            {
                "run": name,
                **metrics["metrics"],
                "fps": timing["fps"],
                "latency_ms_per_image": timing["latency_ms_per_image"],
                "gpu": timing["gpu"],
                "timing_scope": timing["note"],
                "metric_source": f"results/{name}/metrics.json",
                "speed_source": f"results/{name}/speed.json",
            }
        )
    folder = ROOT / "results" / "yolo_baseline_marko_3050"
    with (folder / "frcnn_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        "Wrote four source-backed rows; GPU hardware and timing scope differ; do not infer architecture-only speed gains."
    )


if __name__ == "__main__":
    main()
