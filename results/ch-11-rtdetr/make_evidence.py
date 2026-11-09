"""CH-11 evidence: RT-DETR runs against the CNN families, from stored results only (CPU, no model).

    python results/ch-11-rtdetr/make_evidence.py                # tables + figures (+ saturation if raw preds exist)
    python results/ch-11-rtdetr/make_evidence.py --tables-only  # tables only, from tracked files

Inputs: results/<run>/{metrics,curve,speed,runtime,config}.json of every compared run; for the query-saturation
table, the git-ignored raw predictions outputs/<run>/val_full_predictions.json (when missing, the tracked
query_saturation.json is reused). Outputs in results/ch-11-rtdetr/: comparison.csv, curves.csv, per_class.csv,
density.csv, convergence.csv, query_saturation.json and figures/{learning_curve,size_density}.png.
All AP values come from movd.evaluate (maxDets 500, val_full headline, val_fast curves).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import itertools

from movd import data  # noqa: E402

OUT = ROOT / "results" / "ch-11-rtdetr"
RTDETR_RUNS = ["rtdetr_baseline", "rtdetr_q500", "rtdetr_lr1e4_wb0", "rtdetr_lr1e4"]
# Runs that changed more than their declared variable: kept and shown, but never used as a clean comparison.
CONFOUNDED = {
    "rtdetr_lr1e4": "confounded: explicit optimizer also kept warmup_bias_lr 0.1 (auto sets 0.0); see results/rtdetr_lr1e4/README.md"
}
# (run, family, label). CNN comparators: the CH-09 YOLO curve (required) and the CH-07/CH-08 runs.
RUNS = [
    ("rtdetr_baseline", "query (RT-DETR-l)", "RT-DETR-l baseline (300 queries)"),
    ("rtdetr_q500", "query (RT-DETR-l)", "RT-DETR-l 500 queries"),
    ("rtdetr_lr1e4_wb0", "query (RT-DETR-l)", "RT-DETR-l lr0 1e-4"),
    (
        "rtdetr_lr1e4",
        "query (RT-DETR-l)",
        "RT-DETR-l lr0 1e-4 + bias warmup 0.1 (confounded)",
    ),
    ("yolo_baseline_marko_3050", "anchor-free (YOLO11s)", "YOLO11s baseline (CH-09)"),
    ("frcnn_baseline", "two-stage (Faster R-CNN)", "Faster R-CNN baseline"),
    (
        "frcnn_rpn_pre3000",
        "two-stage (Faster R-CNN)",
        "Faster R-CNN RPN pre3000 (CH-07 selected)",
    ),
    ("frcnn_roi_pos50", "two-stage (Faster R-CNN)", "Faster R-CNN RoI pos50"),
    ("retina_baseline", "one-stage anchor (RetinaNet)", "RetinaNet baseline"),
    (
        "retina_anchors",
        "one-stage anchor (RetinaNet)",
        "RetinaNet anchors (CH-08 selected)",
    ),
    (
        "retina_focal_gamma15",
        "one-stage anchor (RetinaNet)",
        "RetinaNet focal gamma 1.5",
    ),
]
KEYS = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
YOLO_5070_SPEED = OUT / "yolo11s_speed_rtx5070.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def evaluation(run: str) -> dict | None:
    """The movd.evaluate result dict of a run (torchvision runs nest it under 'val_full')."""
    metrics = load(ROOT / "results" / run / "metrics.json")
    if metrics is None:
        return None
    return metrics.get("val_full", metrics)


def curve(run: str) -> list[dict]:
    return load(ROOT / "results" / run / "curve.json") or []


def train_seconds(run: str) -> float | None:
    runtime = load(ROOT / "results" / run / "runtime.json") or {}
    for key in ("train_seconds_total", "train_seconds", "wall_seconds"):
        if isinstance(runtime.get(key), (int, float)):
            return float(runtime[key])
    return None


def queries(run: str) -> int | None:
    if not run.startswith("rtdetr"):
        return None
    config = load(ROOT / "results" / run / "config.json")
    weights = config["run_config"]["weights"]
    if weights.endswith(".yaml"):
        from movd import rtdetr

        return rtdetr.yaml_queries(ROOT / weights)
    return 300  # official rtdetr-l.pt


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def available() -> list[tuple[str, str, str]]:
    return [r for r in RUNS if evaluation(r[0]) is not None and curve(r[0])]


def comparison_rows(runs) -> list[dict]:
    rows = []
    for run, family, label in runs:
        result = evaluation(run)
        points = curve(run)
        fast = [p["val_fast"]["mAP"] for p in points]
        speed = load(ROOT / "results" / run / "speed.json") or {}
        speed = speed.get(
            "inference", speed
        )  # torchvision runs nest the benchmark under "inference"
        row = {
            "run": run,
            "family": family,
            "label": label,
            "caveat": CONFOUNDED.get(run, ""),
        }
        row.update({f"val_full_{k}": result["metrics"][k] for k in KEYS})
        row["val_full_mAP_maxdets100"] = result.get("metrics_maxdets100", {}).get("mAP")
        row.update(
            val_fast_final_mAP=fast[-1],
            val_fast_best_mAP=max(fast),
            val_fast_best_epoch=points[fast.index(max(fast))]["epoch"],
            epochs=len(points),
            queries=queries(run),
            fps=speed.get("fps"),
            latency_ms=speed.get("latency_ms_per_image"),
            speed_gpu=speed.get("gpu"),
            speed_scope=speed.get("note"),
            train_seconds=train_seconds(run),
        )
        rows.append(row)
    yolo_5070 = load(YOLO_5070_SPEED)
    if yolo_5070:
        for row in rows:
            if row["run"] == "yolo_baseline_marko_3050":
                row["fps_rtx5070_same_benchmark"] = yolo_5070["fps"]
    for row in rows:
        if row["run"].startswith("rtdetr"):
            row["fps_rtx5070_same_benchmark"] = row["fps"]
    return rows


def curve_rows(runs) -> list[dict]:
    return [
        {
            "run": run,
            "epoch": p["epoch"],
            "images_seen": p["epoch"] * 2800,
            **{f"val_fast_{k}": p["val_fast"][k] for k in ("mAP", "AP50", "AP_small")},
        }
        for run, _, _ in runs
        for p in curve(run)
    ]


def convergence_rows(runs) -> list[dict]:
    rows = []
    for run, family, _ in runs:
        fast = [p["val_fast"]["mAP"] for p in curve(run)]
        best = max(fast)
        rows.append(
            {
                "run": run,
                "family": family,
                "epoch1_mAP": fast[0],
                "epoch1_share_of_best": fast[0] / best,
                "first_epoch_within_90pct_of_best": next(
                    i + 1 for i, v in enumerate(fast) if v >= 0.9 * best
                ),
                "best_epoch": fast.index(best) + 1,
                "final_minus_best": fast[-1] - best,
                "monotonic_non_decreasing": all(
                    b >= a for a, b in itertools.pairwise(fast)
                ),
            }
        )
    return rows


def per_class_rows(runs) -> list[dict]:
    rows = []
    for run, family, _ in runs:
        for row in evaluation(run)["per_class"]:
            rows.append({"run": run, "family": family, **row})
    return rows


def density_rows(runs) -> list[dict]:
    rows = []
    for run, family, _ in runs:
        for row in evaluation(run)["density"]:
            rows.append({"run": run, "family": family, **row})
    return rows


def query_saturation() -> dict:
    """Per RT-DETR run: detections per val_full image vs the query cap, overall and by target-density bin."""
    path = OUT / "query_saturation.json"
    stored = load(path) or {}
    gt = data.load_coco("val")
    targets = Counter({img["id"]: 0 for img in gt["images"]})
    for ann in gt["annotations"]:
        if not ann.get("iscrowd", 0):
            targets[ann["image_id"]] += 1
    bins = [(0, 20), (20, 50), (50, 100), (100, 10**9)]
    for run in RTDETR_RUNS:
        raw = ROOT / "outputs" / run / "val_full_predictions.json"
        if not raw.exists():
            continue
        cap = queries(run)
        dets = json.loads(raw.read_text(encoding="utf-8"))
        per_image = Counter(d["image_id"] for d in dets)
        ids = list(targets)
        entry = {
            "queries": cap,
            "images": len(ids),
            "at_cap": sum(per_image[i] >= cap for i in ids),
        }
        for low, high in bins:
            sel = [i for i in ids if low <= targets[i] < high]
            label = f"{low}-{high - 1}" if high < 10**9 else f">={low}"
            entry[f"bin_{label}"] = {
                "images": len(sel),
                "mean_targets": sum(targets[i] for i in sel) / max(len(sel), 1),
                "mean_dets": sum(per_image[i] for i in sel) / max(len(sel), 1),
                "share_at_cap": sum(per_image[i] >= cap for i in sel)
                / max(len(sel), 1),
            }
        # Detections with score >= 0.3 per image, a rough count of confident boxes vs the targets.
        conf = Counter(d["image_id"] for d in dets if d["score"] >= 0.3)
        entry["mean_conf03_dets_per_target"] = sum(conf[i] for i in ids) / max(
            sum(targets[i] for i in ids), 1
        )
        stored[run] = entry
    if stored:
        path.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")
    return stored


def figures(runs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    style = {
        "rtdetr_baseline": {"color": "#1f77b4", "marker": "o", "lw": 2.2},
        "rtdetr_q500": {"color": "#17becf", "marker": "s", "lw": 2.2},
        "rtdetr_lr1e4_wb0": {"color": "#9467bd", "marker": "^", "lw": 2.2},
        "rtdetr_lr1e4": {"color": "#c5b0d5", "marker": "v", "ls": ":", "lw": 1.6},
        "yolo_baseline_marko_3050": {"color": "#d62728", "marker": "D", "lw": 2.2},
        "frcnn_rpn_pre3000": {"color": "#2ca02c", "marker": ".", "ls": "--", "lw": 1.4},
        "retina_anchors": {"color": "#ff7f0e", "marker": ".", "ls": "--", "lw": 1.4},
    }
    labels = {run: label for run, _, label in runs}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run, look in style.items():
        if run not in labels:
            continue
        points = curve(run)
        epochs = [p["epoch"] for p in points]
        axes[0].plot(
            epochs, [p["val_fast"]["mAP"] for p in points], label=labels[run], **look
        )
        axes[1].plot(
            epochs,
            [p["val_fast"]["AP_small"] for p in points],
            label=labels[run],
            **look,
        )
    for axis, name in zip(axes, ("mAP@[.5:.95]", "AP_small")):
        axis.set_xlabel("epoch (2,800 train_rubric images each, imgsz/long side 1344)")
        axis.set_ylabel(f"val_fast {name} (movd.evaluate, maxDets 500)")
        axis.grid(alpha=0.3)
    axes[0].set_title("Learning curves: query-based vs CNN detectors")
    axes[1].set_title("Small objects")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "learning_curve.png", dpi=130)
    plt.close(fig)

    shown = [
        r
        for r in (
            "rtdetr_baseline",
            "rtdetr_q500",
            "rtdetr_lr1e4_wb0",
            "yolo_baseline_marko_3050",
            "frcnn_rpn_pre3000",
            "retina_anchors",
        )
        if r in labels
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    width = 0.8 / max(len(shown), 1)
    for k, run in enumerate(shown):
        result = evaluation(run)
        sizes = [result["metrics"][s] for s in ("AP_small", "AP_medium", "AP_large")]
        axes[0].bar(
            [i + k * width for i in range(3)],
            sizes,
            width,
            label=labels[run],
            color=style[run]["color"],
        )
        dens = result["density"]
        axes[1].bar(
            [i + k * width for i in range(len(dens))],
            [d["mAP"] for d in dens],
            width,
            label=labels[run],
            color=style[run]["color"],
        )
        bin_names = [d["bin"] for d in dens]
    axes[0].set_xticks(
        [i + 0.4 - width / 2 for i in range(3)], ["small", "medium", "large"]
    )
    axes[0].set_ylabel("val_full AP@[.5:.95]")
    axes[0].set_title("AP by COCO object size")
    axes[1].set_xticks(
        [i + 0.4 - width / 2 for i in range(len(bin_names))],
        [f"{b} targets" for b in bin_names],
    )
    axes[1].set_ylabel("val_full mAP")
    axes[1].set_title("mAP by targets per frame")
    for axis in axes:
        axis.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "size_density.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    runs = available()
    missing = [r for r, _, _ in RUNS if r not in {x[0] for x in runs}]
    write_csv(OUT / "comparison.csv", comparison_rows(runs))
    write_csv(OUT / "curves.csv", curve_rows(runs))
    write_csv(OUT / "convergence.csv", convergence_rows(runs))
    write_csv(OUT / "per_class.csv", per_class_rows(runs))
    write_csv(OUT / "density.csv", density_rows(runs))
    if not args.tables_only:
        query_saturation()
        figures(runs)
    print(f"{len(runs)} runs; missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
