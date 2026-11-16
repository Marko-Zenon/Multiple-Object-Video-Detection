"""CH-12: aggregate every full run of the four detector families into one comparable table (CPU, stored files only).

    python scripts/aggregate_results.py              # tables + selection + figures
    python scripts/aggregate_results.py --tables-only

Inputs: results/<run>/{config,metrics,curve,runtime,hardware,speed}.json of every run in RUNS (torchvision CH-07/CH-08 and
Ultralytics CH-09/CH-10/CH-11 layouts), the CH-10 replicate spread, and the CH-12 same-session speed benchmark
(results/ch-12-comparison/speed_same_session/summary.json, model_size.json; written by
results/ch-12-comparison/speed_session.py on the GPU).

Outputs:
* results/summary.csv: one row per run with the same metrics, the compatibility verdict and the hardware/batch metadata.
* results/ch-12-comparison/: compatibility.csv (every check per run), per_class.csv, density.csv, curves.csv,
  secondary_checks.csv, selection.json (rule: selection_rule.md) and figures/.

Every AP comes from movd.evaluate (val_full, maxDets 500); the script never re-evaluates. A run enters the primary comparison
only if all compatibility checks pass. Others are listed with the failed check named, never silently mixed in.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from movd import data  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "ch-12-comparison"
KEYS = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
SECONDARY = ["AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
FAMILIES = {
    "frcnn": "two-stage anchor-based (Faster R-CNN R50-FPN v2)",
    "retina": "one-stage anchor-based (RetinaNet R50-FPN v2)",
    "yolo11": "one-stage anchor-free (YOLO11s)",
    "rtdetr": "query-based (RT-DETR-l)",
}
# (run, family key, chunk, role, label). role "selected" = the run the family's own chunk selected at the frozen budget.
RUNS = [
    ("frcnn_baseline", "frcnn", "CH-07", "baseline", "Faster R-CNN baseline"),
    (
        "frcnn_rpn_pre3000",
        "frcnn",
        "CH-07",
        "selected",
        "Faster R-CNN RPN pre-NMS 3000",
    ),
    (
        "frcnn_roi_pos50",
        "frcnn",
        "CH-07",
        "variant",
        "Faster R-CNN RoI positive fraction 0.5",
    ),
    ("retina_baseline", "retina", "CH-08", "baseline", "RetinaNet baseline"),
    ("retina_anchors", "retina", "CH-08", "selected", "RetinaNet data-driven anchors"),
    ("retina_focal_gamma15", "retina", "CH-08", "variant", "RetinaNet focal gamma 1.5"),
    (
        "yolo_baseline_marko_3050",
        "yolo11",
        "CH-09",
        "baseline",
        "YOLO11s baseline (CH-09)",
    ),
    (
        "yolo_baseline_5070",
        "yolo11",
        "CH-10",
        "replicate",
        "YOLO11s baseline replicate",
    ),
    ("yolo_p2", "yolo11", "CH-10", "selected", "YOLO11s + P2 head"),
    (
        "yolo_imgsz1600",
        "yolo11",
        "CH-10",
        "resolution_exception",
        "YOLO11s imgsz 1600 (CH-10 selected, exception)",
    ),
    (
        "rtdetr_baseline",
        "rtdetr",
        "CH-11",
        "selected",
        "RT-DETR-l baseline (300 queries)",
    ),
    ("rtdetr_q500", "rtdetr", "CH-11", "variant", "RT-DETR-l 500 queries"),
    ("rtdetr_lr1e4_wb0", "rtdetr", "CH-11", "variant", "RT-DETR-l lr0 1e-4"),
    (
        "rtdetr_lr1e4",
        "rtdetr",
        "CH-11",
        "confounded",
        "RT-DETR-l lr0 1e-4 + bias warmup 0.1 (confounded)",
    ),
]
CONFOUNDED = {
    "rtdetr_lr1e4": "explicit optimizer also kept warmup_bias_lr 0.1 (auto sets 0.0); results/rtdetr_lr1e4/README.md"
}
EXCEPTIONS = {
    "yolo_imgsz1600": "imgsz 1600: approved CH-10 resolution exception, not the frozen 1344"
}
REPLICATE_PAIR = ("yolo_baseline_marko_3050", "yolo_baseline_5070")
TORCHVISION = {"frcnn", "retina"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_file(run: str, name: str):
    return load(RESULTS / run / f"{name}.json")


def evaluation(run: str) -> dict:
    metrics = run_file(run, "metrics")
    return metrics.get(
        "val_full", metrics
    )  # torchvision runs nest the result under "val_full"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metadata(run: str, family: str) -> dict:
    """Training/evaluation settings in one schema, from each framework's own stored files."""
    config = run_file(run, "config")
    runtime = run_file(run, "runtime") or {}
    hardware = run_file(run, "hardware") or {}
    result = evaluation(run)
    if family in TORCHVISION:
        images_seen = runtime.get("images_seen")
        return {
            "framework": "torchvision",
            "train_split": config["train_split"],
            "epochs": config["epochs"],
            "images_seen": images_seen,
            "input": f"min {config['min_size']} / max {config['max_size']}",
            "long_side": config["max_size"],
            "batch_size": config["batch_size"],
            "seed": config["seed"],
            "amp": config["amp"],
            "pretrained": f"torchvision {config['weights']}",
            "model_config": json.dumps(
                {
                    k: config[k]
                    for k in (
                        "learning_rate",
                        "rpn",
                        "roi",
                        "anchors",
                        "focal_alpha",
                        "focal_gamma",
                    )
                    if k in config
                },
                sort_keys=True,
            ),
            "checkpoint": "final epoch (epoch_0006.pt), evaluated at the end of training",
            "train_gpu": hardware.get("gpu"),
            "train_seconds": runtime.get("train_seconds"),
            "eval_split": result.get("split"),
        }
    kwargs = config["ultralytics_train_kwargs"]
    processes = runtime.get("training_processes", [])
    batches = sum(p.get("batches", 0) for p in processes)
    samples = {p.get("samples") for p in processes}
    return {
        "framework": "ultralytics",
        "train_split": config["train_split"],
        "epochs": kwargs["epochs"],
        # observed batches (batch 1) when recorded; otherwise epochs x exported split size
        "images_seen": batches * kwargs["batch"]
        if batches
        else kwargs["epochs"] * config["export_manifest"]["num_images"],
        "input": f"imgsz {kwargs['imgsz']}",
        "long_side": kwargs["imgsz"],
        "batch_size": kwargs["batch"],
        "seed": kwargs["seed"],
        "amp": kwargs["amp"],
        "pretrained": f"{Path(config['weights']).name} sha256 {config['weights_sha256'][:12]}",
        "model_config": json.dumps(
            {
                k: v
                for k, v in config["run_config"].get("overrides", {}).items()
                if k != "cache"
            },
            sort_keys=True,
        ),
        "checkpoint": result.get("checkpoint"),
        "train_gpu": hardware.get("gpu"),
        "train_seconds": runtime.get("train_seconds_total"),
        "eval_split": result.get("split"),
        "train_samples_per_epoch": ",".join(str(s) for s in sorted(samples, key=str))
        if samples
        else None,
    }


def compatibility(run: str, family: str, meta: dict) -> dict:
    protocol = data.load_protocol()
    budget = protocol["rubric_budget"]
    result = evaluation(run)
    checks = {
        "eval_split_val_full": meta["eval_split"] == "val_full",
        "eval_frames_2846": result["num_images"]
        == len(data.split_image_ids("val_full")),
        "no_predictions_outside_split": result["num_predictions_outside_split"] == 0,
        "max_dets_500": result["max_dets"] == 500
        and result["images_over_headline_max_dets"] == 0,
        "ignore_policy_crowd_kept": "iscrowd=1 copies kept"
        in result["protocol"]["ignore_regions"],
        "train_split_rubric": meta["train_split"] == "train_rubric",
        "epochs_6": meta["epochs"] == budget["epochs"],
        "images_seen_16800": meta["images_seen"] == budget["epochs"] * 2800,
        "long_side_1344": meta["long_side"] == budget["input_long_side"],
        "seed_42": meta["seed"] == protocol["seed"],
        "single_variable_clean": run not in CONFOUNDED,
        "hardware_recorded": bool(meta["train_gpu"]),
        "checkpoint_recorded": bool(meta["checkpoint"]),
    }
    failed = [k for k, ok in checks.items() if not ok]
    reason = "; ".join(filter(None, [CONFOUNDED.get(run), EXCEPTIONS.get(run)]))
    return {
        "run": run,
        **checks,
        "comparable": not failed,
        "failed_checks": ",".join(failed),
        "reason": reason,
    }


def recorded_hashes() -> dict[str, tuple[str, str]]:
    """Checkpoint SHA-256 values recorded by earlier chunks: run -> (sha256, source file)."""
    known = {}
    ch10 = load(
        RESULTS / "ch-10-yolo-tuning" / "speed_same_session" / "summary.json"
    ) or {"runs": {}}
    for run, entry in ch10["runs"].items():
        known[run] = (
            entry["checkpoint_sha256"],
            "results/ch-10-yolo-tuning/speed_same_session/summary.json",
        )
    ch11 = load(RESULTS / "ch-11-rtdetr" / "yolo11s_speed_rtx5070.json") or {}
    if ch11.get("weights_sha256"):
        known.setdefault(
            "yolo_baseline_marko_3050",
            (ch11["weights_sha256"], "results/ch-11-rtdetr/yolo11s_speed_rtx5070.json"),
        )
    with open(RESULTS / "submissions.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "frcnn_rpn_pre3000/epoch_0006.pt" in row["checkpoint"]:
                known["frcnn_rpn_pre3000"] = (
                    row["checkpoint_sha256"],
                    "results/submissions.csv (CH-05 upload)",
                )
    return known


def speed_rows() -> dict:
    summary = load(OUT / "speed_same_session" / "summary.json") or {"runs": {}}
    sizes = load(OUT / "model_size.json") or {}
    known = recorded_hashes()
    rows = {}
    for run in [r[0] for r in RUNS]:
        entry = summary["runs"].get(run, {})
        sha, source = known.get(run, (None, None))
        if entry.get("checkpoint_sha256") and sha:
            identity = (
                f"matches {source}"
                if sha == entry["checkpoint_sha256"]
                else f"MISMATCH with {source}"
            )
        elif entry.get("checkpoint_sha256"):
            identity = "hashed in the CH-12 session; no earlier record"
        else:
            identity = "not re-timed or hashed in CH-12"
        own = run_file(run, "speed") or {}
        own = own.get("inference", own)
        rows[run] = {
            "fps_common_rtx5070_same_session": entry.get("common", {}).get(
                "fps_median"
            ),
            "fps_native_tv_rtx5070_same_session": entry.get("native_tv", {}).get(
                "fps_median"
            ),
            "checkpoint_sha256": entry.get("checkpoint_sha256"),
            "checkpoint_identity": identity,
            "parameters": sizes.get(run, {}).get("parameters"),
            "fps_own_run": own.get("fps"),
            "fps_own_run_gpu": own.get("gpu"),
            "fps_own_run_scope": own.get("note"),
        }
    return rows


def summary_rows() -> tuple[list[dict], list[dict]]:
    speeds = speed_rows()
    rows, compat = [], []
    for run, family, chunk, role, label in RUNS:
        meta = metadata(run, family)
        check = compatibility(run, family, meta)
        compat.append(check)
        result = evaluation(run)
        fast = [p["val_fast"]["mAP"] for p in run_file(run, "curve")]
        row = {
            "run": run,
            "family": FAMILIES[family],
            "chunk": chunk,
            "role": role,
            "label": label,
            "comparable": check["comparable"],
            "failed_checks": check["failed_checks"],
            "reason": check["reason"],
        }
        row.update({f"val_full_{k}": result["metrics"][k] for k in KEYS})
        row["val_full_mAP_maxdets100"] = result["metrics_maxdets100"]["mAP"]
        row.update(
            val_fast_final_mAP=fast[-1],
            val_fast_best_mAP=max(fast),
            val_fast_best_epoch=fast.index(max(fast)) + 1,
        )
        row["num_predictions"] = result["num_predictions"]
        row.update({k: v for k, v in meta.items() if k not in ("eval_split",)})
        row.update(speeds[run])
        rows.append(row)
    return rows, compat


def per_class_rows() -> list[dict]:
    return [
        {"run": run, "label": label, "family": family, **c}
        for run, family, _, _, label in RUNS
        for c in evaluation(run)["per_class"]
    ]


def density_rows() -> list[dict]:
    return [
        {"run": run, "label": label, "family": family, **d}
        for run, family, _, _, label in RUNS
        for d in evaluation(run)["density"]
    ]


def curve_rows() -> list[dict]:
    return [
        {
            "run": run,
            "family": family,
            "epoch": p["epoch"],
            "images_seen": p["epoch"] * 2800,
            **{f"val_fast_{k}": p["val_fast"][k] for k in ("mAP", "AP50", "AP_small")},
        }
        for run, family, _, _, _ in RUNS
        for p in run_file(run, "curve")
    ]


# Failure cases of each family's selected run, all found with the CH-07 rule (movd.ultra.find_failure_cases).
FAILURE_SOURCES = {
    "frcnn_rpn_pre3000": (
        "results/ch-07-frcnn/failure_cases.csv",
        "results/ch-07-frcnn/figures",
    ),
    "retina_anchors": (
        "results/ch-12-comparison/retina_anchors_failures/failure_cases.csv",
        "results/ch-12-comparison/retina_anchors_failures/figures",
    ),
    "yolo_p2": ("results/yolo_p2/failure_cases.csv", "results/yolo_p2/figures"),
    "rtdetr_baseline": (
        "results/rtdetr_baseline/failure_cases.csv",
        "results/rtdetr_baseline/figures",
    ),
}


def failure_rows() -> list[dict]:
    family = {r[0]: r[1] for r in RUNS}
    rows = []
    for run, (table, figures_dir) in FAILURE_SOURCES.items():
        with open(ROOT / table, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "run": run,
                        "family": FAMILIES[family[run]],
                        "case": row["case"],
                        "frame": row["frame"],
                        "target": row["target"],
                        "target_size_px": f"{row['target_width_px']}x{row['target_height_px']}",
                        "frame_targets": row["frame_targets"],
                        "observation": row["observation"],
                        "figure": f"{figures_dir}/{row['case']}.png",
                        "source": table,
                    }
                )
    return rows


def selection(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Apply results/ch-12-comparison/selection_rule.md to the comparable family-selected runs."""
    by_run = {r["run"]: r for r in rows}
    finalists = [r for r in rows if r["role"] == "selected" and r["comparable"]]
    ranked = sorted(finalists, key=lambda r: r["val_full_mAP"], reverse=True)
    leader, runner_up = ranked[0], ranked[1]
    spread = abs(
        by_run[REPLICATE_PAIR[1]]["val_full_mAP"]
        - by_run[REPLICATE_PAIR[0]]["val_full_mAP"]
    )
    density = {(d["run"], d["bin"]): d["mAP"] for d in density_rows()}
    bins = [d["bin"] for d in evaluation(leader["run"])["density"]]
    checks = []
    for other in ranked[1:]:
        for key in SECONDARY:
            checks.append(
                {
                    "leader": leader["run"],
                    "other": other["run"],
                    "check": key,
                    "leader_value": leader[f"val_full_{key}"],
                    "other_value": other[f"val_full_{key}"],
                    "leader_ahead": leader[f"val_full_{key}"]
                    > other[f"val_full_{key}"],
                }
            )
        for b in bins:
            checks.append(
                {
                    "leader": leader["run"],
                    "other": other["run"],
                    "check": f"density {b}",
                    "leader_value": density[(leader["run"], b)],
                    "other_value": density[(other["run"], b)],
                    "leader_ahead": density[(leader["run"], b)]
                    > density[(other["run"], b)],
                }
            )
    vs_runner = [c for c in checks if c["other"] == runner_up["run"]]
    lead = leader["val_full_mAP"] - runner_up["val_full_mAP"]
    secondary_ok = sum(c["leader_ahead"] for c in vs_runner) > len(vs_runner) / 2
    tie = lead <= spread or not secondary_ok
    exception = by_run["yolo_imgsz1600"]
    result = {
        "rule": "results/ch-12-comparison/selection_rule.md",
        "primary_metric": "val_full mAP@[.5:.95], movd.evaluate maxDets 500, comparable runs only",
        "family_representatives": {r["run"]: r["val_full_mAP"] for r in ranked},
        "leader": leader["run"],
        "runner_up": runner_up["run"],
        "lead_mAP": lead,
        "run_to_run_spread_val_full_mAP": spread,
        "lead_over_spread": lead / spread,
        "secondary_checks_won_vs_runner_up": f"{sum(c['leader_ahead'] for c in vs_runner)}/{len(vs_runner)}",
        "secondary_checks_won_vs_all": f"{sum(c['leader_ahead'] for c in checks)}/{len(checks)}",
        "secondary_checks_lost": [
            f"{c['check']} vs {c['other']}" for c in checks if not c["leader_ahead"]
        ],
        "tie": tie,
        "finalist_family": None if tie else leader["family"],
        "finalist_run": None if tie else leader["run"],
        "resolution_exception": {
            "run": "yolo_imgsz1600",
            "val_full_mAP": exception["val_full_mAP"],
            "below_finalist_by": leader["val_full_mAP"] - exception["val_full_mAP"],
            "note": "Reported separately (rule 5). Not fixed-resolution evidence. Its family is not the finalist, and the finalist "
            "has no evidence at 1600, so it does not change the final input size.",
        },
        "excluded_from_primary": {
            r["run"]: r["failed_checks"] for r in rows if not r["comparable"]
        },
        "note": "One seed per configuration. Spread from the single CH-10 replicate pair (another GPU).",
    }
    return result, checks


def figures(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "frcnn": "#2ca02c",
        "retina": "#ff7f0e",
        "yolo11": "#d62728",
        "rtdetr": "#1f77b4",
    }
    fam = {r[0]: r[1] for r in RUNS}
    shown = [
        r for r in rows if r["role"] in ("selected", "baseline", "resolution_exception")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for axis, key in zip(axes, ("mAP", "AP50", "AP_small")):
        values = [r[f"val_full_{key}"] for r in shown]
        bars = axis.barh(
            [r["label"] for r in shown],
            values,
            color=[colors[fam[r["run"]]] for r in shown],
        )
        for bar, r in zip(bars, shown):
            if not r["comparable"]:
                bar.set_hatch("//")
                bar.set_alpha(0.6)
        axis.set_title(f"val_full {key}")
        axis.grid(axis="x", alpha=0.3)
        axis.invert_yaxis()
    for axis in axes[1:]:
        axis.set_yticklabels([])
    fig.suptitle(
        "Four families at the rubric budget (hatched: not comparable, see compatibility.csv)"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy.png", dpi=130)
    plt.close(fig)

    timed = [r for r in shown if r["fps_common_rtx5070_same_session"]]
    if timed:
        fig, axis = plt.subplots(figsize=(8, 5.5))
        for r in timed:
            axis.scatter(
                r["fps_common_rtx5070_same_session"],
                r["val_full_mAP"],
                s=80,
                color=colors[fam[r["run"]]],
                marker="o" if r["comparable"] else "x",
            )
            axis.annotate(
                r["label"],
                (r["fps_common_rtx5070_same_session"], r["val_full_mAP"]),
                fontsize=7,
                xytext=(4, 3),
                textcoords="offset points",
            )
        axis.set_xscale("log")
        axis.set_xlabel(
            "FPS, RTX 5070 Laptop, one session, common scope (FP32, host BGR array, batch 1), log scale"
        )
        axis.set_ylabel("val_full mAP@[.5:.95]")
        axis.set_title("Accuracy vs speed (x: not comparable)")
        axis.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(fig_dir / "accuracy_speed.png", dpi=130)
        plt.close(fig)

    reps = [r for r in rows if r["role"] == "selected"] + [
        r for r in rows if r["role"] == "resolution_exception"
    ]
    names = [c["name"] for c in evaluation(reps[0]["run"])["per_class"]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    matrix = [[c["AP"] for c in evaluation(r["run"])["per_class"]] for r in reps]
    image = axes[0].imshow(matrix, cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(len(names)), names, rotation=30)
    axes[0].set_yticks(range(len(reps)), [r["label"] for r in reps], fontsize=8)
    for i, line in enumerate(matrix):
        for j, value in enumerate(line):
            axes[0].text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="w" if value < 0.3 else "k",
                fontsize=7,
            )
    axes[0].set_title("val_full AP per class")
    fig.colorbar(image, ax=axes[0])
    bins = [d["bin"] for d in evaluation(reps[0]["run"])["density"]]
    width = 0.8 / len(reps)
    for k, r in enumerate(reps):
        axes[1].bar(
            [i + k * width for i in range(len(bins))],
            [d["mAP"] for d in evaluation(r["run"])["density"]],
            width,
            label=r["label"],
            color=colors[fam[r["run"]]],
            hatch=None if r["comparable"] else "//",
            alpha=1 if r["comparable"] else 0.6,
        )
    axes[1].set_xticks(
        [i + 0.4 - width / 2 for i in range(len(bins))], [f"{b} targets" for b in bins]
    )
    axes[1].set_ylabel("val_full mAP")
    axes[1].set_title("Dense scenes: mAP by targets per frame")
    axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "per_class_density.png", dpi=130)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for r in reps:
        points = run_file(r["run"], "curve")
        axis.plot(
            [p["epoch"] for p in points],
            [p["val_fast"]["mAP"] for p in points],
            marker="o",
            color=colors[fam[r["run"]]],
            ls="-" if r["comparable"] else ":",
            label=r["label"],
        )
    axis.set_xlabel("epoch (2,800 train_rubric images each)")
    axis.set_ylabel("val_fast mAP (movd.evaluate, maxDets 500)")
    axis.set_title("Learning curves of the family representatives")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "learning_curves.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, compat = summary_rows()
    write_csv(RESULTS / "summary.csv", rows)
    write_csv(OUT / "compatibility.csv", compat)
    write_csv(OUT / "per_class.csv", per_class_rows())
    write_csv(OUT / "density.csv", density_rows())
    write_csv(OUT / "curves.csv", curve_rows())
    write_csv(OUT / "failures.csv", failure_rows())
    mismatched = [
        r["run"] for r in rows if r["checkpoint_identity"].startswith("MISMATCH")
    ]
    if mismatched:
        raise SystemExit(f"checkpoint hash mismatch: {mismatched}")
    chosen, checks = selection(rows)
    write_csv(OUT / "secondary_checks.csv", checks)
    (OUT / "selection.json").write_text(
        json.dumps(chosen, indent=2) + "\n", encoding="utf-8"
    )
    if not args.tables_only:
        figures(rows)
    print(
        f"{len(rows)} runs, {sum(r['comparable'] for r in rows)} comparable; finalist: {chosen['finalist_run']} "
        f"(lead {chosen['lead_mAP']:.4f} = {chosen['lead_over_spread']:.1f}x spread, "
        f"secondary vs runner-up {chosen['secondary_checks_won_vs_runner_up']})"
    )


if __name__ == "__main__":
    main()
