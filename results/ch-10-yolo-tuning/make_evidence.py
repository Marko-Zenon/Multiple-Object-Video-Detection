"""CH-10 evidence: YOLO11s variants vs the CH-09 baseline, from stored results (CPU, no model).

    python results/ch-10-yolo-tuning/make_evidence.py                # tables + figures (+ fine sizes if raw preds exist)
    python results/ch-10-yolo-tuning/make_evidence.py --tables-only  # tables only, from tracked files

Inputs: results/<run>/{metrics,curve,speed,runtime,config}.json of every run, the same-session speed files in
speed_same_session/, and for the fine size buckets the git-ignored raw predictions
outputs/<run>/val_full_predictions.json (when missing, the tracked fine_size.csv rows are kept).
Outputs in results/ch-10-yolo-tuning/: comparison.csv, size.csv, fine_size.csv, density.csv, per_class.csv,
curves.csv, failures.csv, selection.json and figures/{learning_curve,size_density,fine_size,accuracy_speed}.png.
All AP values come from movd.evaluate (maxDets 500, val_full headline, val_fast curves); fine_size.csv uses
the same GT, crowd handling and 500 cap with finer area ranges (supplementary, not a headline metric).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from movd import data  # noqa: E402

OUT = ROOT / "results" / "ch-10-yolo-tuning"
BASELINE = "yolo_baseline_marko_3050"
REPLICATE = "yolo_baseline_5070"
# (run, label, role). role: reference / replicate / variant.
RUNS = [
    (BASELINE, "YOLO11s baseline (CH-09, RTX 3050)", "reference"),
    (REPLICATE, "YOLO11s baseline replicate (RTX 5070)", "replicate"),
    ("yolo_p2", "YOLO11s + P2 head (1344)", "variant"),
    ("yolo_imgsz1600", "YOLO11s imgsz 1600 (resolution exception)", "variant"),
]
CANDIDATES = [BASELINE, "yolo_p2", "yolo_imgsz1600"]
KEYS = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
# Finer area ranges (px^2) inside COCO "small" (< 32^2); medium and large as COCO.
FINE_BUCKETS = [
    ("<8px", 0, 8**2),
    ("8-16px", 8**2, 16**2),
    ("16-32px", 16**2, 32**2),
    ("32-96px", 32**2, 96**2),
    (">96px", 96**2, 1e10),
]
SPEED_DIR = OUT / "speed_same_session"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def evaluation(run: str) -> dict | None:
    return load(ROOT / "results" / run / "metrics.json")


def curve(run: str) -> list[dict]:
    return load(ROOT / "results" / run / "curve.json") or []


def config(run: str) -> dict:
    return load(ROOT / "results" / run / "config.json")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def available() -> list[tuple[str, str, str]]:
    return [r for r in RUNS if evaluation(r[0]) is not None and len(curve(r[0])) == 6]


def training(run: str) -> dict:
    runtime = load(ROOT / "results" / run / "runtime.json") or {}
    processes = runtime.get("training_processes", [])
    return {
        "train_seconds": runtime.get("train_seconds_total"),
        "train_processes": len(runtime.get("train_process_seconds", [])),
        "peak_cuda_allocated_gb": round(
            max(p.get("peak_cuda_allocated_bytes", 0) for p in processes) / 1e9, 3
        )
        if processes
        else None,
        "train_gpu": (load(ROOT / "results" / run / "hardware.json") or {}).get("gpu"),
    }


def same_session_speed() -> dict:
    """Median FPS per run over the interleaved rounds in speed_same_session/ (one RTX 5070 session)."""
    summary = load(SPEED_DIR / "summary.json")
    return summary["runs"] if summary else {}


def comparison_rows(runs) -> list[dict]:
    rows, speeds = [], same_session_speed()
    base = evaluation(BASELINE)["metrics"]
    for run, label, role in runs:
        result = evaluation(run)
        fast = [p["val_fast"]["mAP"] for p in curve(run)]
        settings = config(run)["ultralytics_train_kwargs"]
        own_speed = load(ROOT / "results" / run / "speed.json") or {}
        row = {"run": run, "label": label, "role": role, "imgsz": settings["imgsz"]}
        row["weights"] = config(run)["run_config"]["weights"]
        row.update({f"val_full_{k}": result["metrics"][k] for k in KEYS})
        row.update(
            {f"delta_{k}_vs_baseline": result["metrics"][k] - base[k] for k in KEYS}
        )
        row["val_full_mAP_maxdets100"] = result["metrics_maxdets100"]["mAP"]
        row.update(
            val_fast_final_mAP=fast[-1],
            val_fast_best_mAP=max(fast),
            val_fast_best_epoch=fast.index(max(fast)) + 1,
            num_predictions=result["num_predictions"],
            checkpoint=result["checkpoint"],
        )
        row.update(training(run))
        row.update(
            fps_own_run=own_speed.get("fps"),
            fps_own_run_gpu=own_speed.get("gpu"),
            fps_rtx5070_same_session=speeds.get(run, {}).get("fps_median"),
            latency_ms_rtx5070_same_session=speeds.get(run, {}).get(
                "latency_ms_median"
            ),
        )
        rows.append(row)
    return rows


def size_rows(runs) -> list[dict]:
    base = evaluation(BASELINE)["metrics"]
    rows = []
    for run, label, _ in runs:
        m = evaluation(run)["metrics"]
        for size in ("small", "medium", "large"):
            key = f"AP_{size}"
            rows.append(
                {
                    "run": run,
                    "label": label,
                    "size": size,
                    "AP": m[key],
                    "delta_vs_baseline": m[key] - base[key],
                }
            )
    return rows


def density_rows(runs) -> list[dict]:
    base = {d["bin"]: d for d in evaluation(BASELINE)["density"]}
    rows = []
    for run, label, _ in runs:
        for d in evaluation(run)["density"]:
            rows.append(
                {
                    "run": run,
                    "label": label,
                    **d,
                    "delta_mAP_vs_baseline": d["mAP"] - base[d["bin"]]["mAP"],
                    "maxdets500_minus_100": d["mAP"] - d["mAP_maxdets100"],
                }
            )
    return rows


def per_class_rows(runs) -> list[dict]:
    base = {c["name"]: c["AP"] for c in evaluation(BASELINE)["per_class"]}
    return [
        {
            "run": run,
            "label": label,
            **c,
            "delta_AP_vs_baseline": c["AP"] - base[c["name"]],
        }
        for run, label, _ in runs
        for c in evaluation(run)["per_class"]
    ]


def curve_rows(runs) -> list[dict]:
    return [
        {
            "run": run,
            "epoch": p["epoch"],
            "images_seen": p["epoch"] * 2800,
            **{f"val_fast_{k}": p["val_fast"][k] for k in ("mAP", "AP50", "AP_small")},
            "num_predictions": p["num_predictions"],
        }
        for run, _, _ in runs
        for p in curve(run)
    ]


def failure_rows(runs) -> list[dict]:
    rows = []
    for run, label, _ in runs:
        for row in read_csv(ROOT / "results" / run / "failure_cases.csv"):
            rows.append(
                {
                    "run": run,
                    "label": label,
                    **row,
                    "figure": f"results/{run}/{row['figure']}",
                }
            )
    return rows


def fine_size_eval(predictions: list[dict]) -> list[dict]:
    """AP@[.5:.95], AP50 and recall (maxDets 500) per fine GT-area bucket, crowd regions handled as in movd.evaluate."""
    from pycocotools.cocoeval import COCOeval

    from movd import evaluate as ev

    gt = data.load_coco("val")
    ids = sorted(data.split_image_ids("val_full"))
    coco_gt = ev._coco_gt(gt)
    kept = [p for p in predictions if p["image_id"] in set(ids)]
    E = COCOeval(
        coco_gt,
        ev._coco_dt(coco_gt, ev.cap_per_image(kept, ev.HEADLINE_MAX_DETS)),
        iouType="bbox",
    )
    E.params.imgIds = ids
    E.params.maxDets = [ev.HEADLINE_MAX_DETS]
    E.params.areaRng = [[low, high] for _, low, high in FINE_BUCKETS]
    E.params.areaRngLbl = [name for name, _, _ in FINE_BUCKETS]
    ev._quiet(E.evaluate)
    ev._quiet(E.accumulate)
    counts = {name: 0 for name, _, _ in FINE_BUCKETS}
    for ann in gt["annotations"]:
        if ann["image_id"] in set(ids) and not ann.get("iscrowd", 0):
            for name, low, high in FINE_BUCKETS:
                if low <= ann["area"] < high:
                    counts[name] += 1
    rows = []
    for a, (name, _, _) in enumerate(FINE_BUCKETS):
        p = E.eval["precision"][:, :, :, a, 0]
        r = E.eval["recall"][:, :, a, 0]
        p50 = E.eval["precision"][ev._iou_index(E, 0.5), :, :, a, 0]
        r50 = E.eval["recall"][ev._iou_index(E, 0.5), :, a, 0]
        rows.append(
            {
                "bucket": name,
                "num_gt": counts[name],
                "AP": float(p[p > -1].mean()),
                "AP50": float(p50[p50 > -1].mean()),
                "AR": float(r[r > -1].mean()),
                "recall_at_IoU50": float(r50[r50 > -1].mean()),
            }
        )
    return rows


def fine_size_rows(runs) -> list[dict]:
    stored = {}
    for row in read_csv(OUT / "fine_size.csv"):
        stored.setdefault(row["run"], []).append(row)
    rows = []
    for run, label, _ in runs:
        raw = ROOT / "outputs" / run / "val_full_predictions.json"
        if raw.exists():
            for row in fine_size_eval(json.loads(raw.read_text(encoding="utf-8"))):
                rows.append({"run": run, "label": label, **row})
        else:
            rows.extend(stored.get(run, []))
    return rows


def selection(runs) -> dict:
    """Apply results/ch-10-yolo-tuning/selection_rule.md (written before the variants finished)."""
    names = {r[0] for r in runs}
    final = {run: curve(run)[-1]["val_fast"]["mAP"] for run in names}
    candidates = [r for r in CANDIDATES if r in names]
    winner = max(candidates, key=final.get)
    spread = abs(final[REPLICATE] - final[BASELINE]) if REPLICATE in names else None
    full = {run: evaluation(run)["metrics"]["mAP"] for run in names}
    at_1344 = [
        r for r in candidates if config(r)["ultralytics_train_kwargs"]["imgsz"] == 1344
    ]
    best_1344 = max(at_1344, key=final.get)
    gain = final[winner] - final[BASELINE]
    return {
        "rule": "results/ch-10-yolo-tuning/selection_rule.md",
        "metric": "val_fast mAP of the epoch-6 checkpoint (last.pt), movd.evaluate maxDets 500",
        "candidates": {r: final[r] for r in candidates},
        "selected": winner,
        "selected_imgsz": config(winner)["ultralytics_train_kwargs"]["imgsz"],
        "selected_gain_over_baseline_val_fast": gain,
        "replicate_spread_val_fast": spread,
        "replicate_spread_val_full": abs(full[REPLICATE] - full[BASELINE])
        if REPLICATE in names
        else None,
        "gain_within_replicate_spread": None
        if spread is None or winner == BASELINE
        else gain <= spread,
        "best_at_frozen_1344": best_1344,
        "val_full_mAP": full,
        "note": "One seed per configuration. val_full is reported for every run but does not select.",
    }


def figures(runs, fine_rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    style = {
        BASELINE: {"color": "#d62728", "marker": "D"},
        REPLICATE: {"color": "#ff9896", "marker": "d", "ls": "--"},
        "yolo_p2": {"color": "#1f77b4", "marker": "s"},
        "yolo_imgsz1600": {"color": "#2ca02c", "marker": "^"},
    }
    labels = {run: label for run, label, _ in runs}
    shown = [r for r, _, _ in runs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for run in shown:
        points = curve(run)
        epochs = [p["epoch"] for p in points]
        for axis, key in zip(axes, ("mAP", "AP_small")):
            axis.plot(
                epochs,
                [p["val_fast"][key] for p in points],
                label=labels[run],
                lw=2,
                **style[run],
            )
    for axis, name in zip(axes, ("mAP@[.5:.95]", "AP_small")):
        axis.set_xlabel("epoch (2,800 train_rubric images each)")
        axis.set_ylabel(f"val_fast {name} (movd.evaluate, maxDets 500)")
        axis.grid(alpha=0.3)
    axes[0].set_title("YOLO11s variants: learning curves")
    axes[1].set_title("Small objects")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(fig_dir / "learning_curve.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    width = 0.8 / len(shown)
    for k, run in enumerate(shown):
        result = evaluation(run)
        sizes = [result["metrics"][f"AP_{s}"] for s in ("small", "medium", "large")]
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
            color=style[run]["color"],
        )
    axes[0].set_xticks(
        [i + 0.4 - width / 2 for i in range(3)], ["small", "medium", "large"]
    )
    axes[0].set_ylabel("val_full AP@[.5:.95]")
    axes[0].set_title("AP by COCO object size")
    bins = [d["bin"] for d in evaluation(BASELINE)["density"]]
    axes[1].set_xticks(
        [i + 0.4 - width / 2 for i in range(len(bins))], [f"{b} targets" for b in bins]
    )
    axes[1].set_ylabel("val_full mAP")
    axes[1].set_title("mAP by targets per frame")
    for axis in axes:
        axis.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "size_density.png", dpi=130)
    plt.close(fig)

    if fine_rows:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        buckets = [b for b, _, _ in FINE_BUCKETS]
        for k, run in enumerate(shown):
            rows = {r["bucket"]: r for r in fine_rows if r["run"] == run}
            if not rows:
                continue
            for axis, key in zip(axes, ("AP", "AR")):
                axis.bar(
                    [i + k * width for i in range(len(buckets))],
                    [float(rows[b][key]) for b in buckets],
                    width,
                    label=labels[run],
                    color=style[run]["color"],
                )
        counts = {r["bucket"]: r["num_gt"] for r in fine_rows if r["run"] == BASELINE}
        for axis, key in zip(axes, ("AP@[.5:.95]", "recall (AR, maxDets 500)")):
            axis.set_xticks(
                [i + 0.4 - width / 2 for i in range(len(buckets))],
                [f"{b}\n(n={counts.get(b, '?')})" for b in buckets],
            )
            axis.set_xlabel("GT box size, sqrt(area) in original pixels")
            axis.set_ylabel(f"val_full {key}")
            axis.grid(axis="y", alpha=0.3)
        axes[0].set_title("AP by finer object size (supplementary)")
        axes[1].set_title("Recall by finer object size")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "fine_size.png", dpi=130)
        plt.close(fig)

    speeds = same_session_speed()
    if speeds:
        fig, axis = plt.subplots(figsize=(6.5, 4.5))
        for run in shown:
            if run in speeds:
                axis.scatter(
                    speeds[run]["fps_median"],
                    evaluation(run)["metrics"]["mAP"],
                    s=70,
                    color=style[run]["color"],
                    marker=style[run]["marker"],
                    label=labels[run],
                )
        axis.set_xlabel(
            "FPS, RTX 5070 Laptop, same session (batch 1, median of rounds)"
        )
        axis.set_ylabel("val_full mAP@[.5:.95]")
        axis.set_title("Accuracy vs speed")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "accuracy_speed.png", dpi=130)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    runs = available()
    missing = [r for r, _, _ in RUNS if r not in {x[0] for x in runs}]
    write_csv(OUT / "comparison.csv", comparison_rows(runs))
    write_csv(OUT / "size.csv", size_rows(runs))
    write_csv(OUT / "density.csv", density_rows(runs))
    write_csv(OUT / "per_class.csv", per_class_rows(runs))
    write_csv(OUT / "curves.csv", curve_rows(runs))
    write_csv(OUT / "failures.csv", failure_rows(runs))
    fine = read_csv(OUT / "fine_size.csv") if args.tables_only else fine_size_rows(runs)
    if not args.tables_only:
        write_csv(OUT / "fine_size.csv", fine)
    (OUT / "selection.json").write_text(
        json.dumps(selection(runs), indent=2) + "\n", encoding="utf-8"
    )
    if not args.tables_only:
        figures(runs, fine)
    print(f"{len(runs)} runs; missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
