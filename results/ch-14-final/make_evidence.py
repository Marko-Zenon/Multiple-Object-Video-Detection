"""Build the CH-14 evidence tables, figures and the rule-B decision from stored results (CPU only).

    python results/ch-14-final/make_evidence.py [--hash-checkpoints]

Reads (nothing is recomputed on the GPU):
* results/final_rtdetr/{curve.json, runtime.json, ultralytics/results.csv, config.json}  (CH-14 run)
* results/rtdetr_baseline/{curve.json, metrics.json}                                       (CH-11 fallback, single pass)
* results/ch-13-inference/runs/val_full/combo_flip_tile_nms05/metrics.json                 (fallback, CH-13 pipeline)
* results/ch-14-final/final_rtdetr/val_full/{single,pipeline}/metrics.json                 (CH-14, both modes)
* results/ch-14-final/speed/session.json                                                   (same-session speed)
Writes to results/ch-14-final/: epochs.csv, comparison.csv, per_class.csv, density.csv, selection.json,
checkpoints.json (with --hash-checkpoints) and figures/{curve,per_class,density}.png.
The decision applies results/ch-14-final/selection_rule.md (committed before any CH-14 result) as written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = REPO / "results"
SPREAD = 0.0063  # CH-10 replicate spread (CH-12 rule 3)
APS_GUARD = 0.003  # CH-13 small-object guard
SECONDARY = ["AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
BLUE, ORANGE = "#2a78d6", "#eb6834"  # dataviz reference palette, categorical slots 1-2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def models() -> dict[str, dict]:
    """The four val_full evaluations: CH-14 C and fallback F, single pass and CH-13 pipeline."""
    c = HERE / "final_rtdetr" / "val_full"
    return {
        "C_single": load(c / "single" / "metrics.json"),
        "C_pipeline": load(c / "pipeline" / "metrics.json"),
        "F_single": load(RES / "rtdetr_baseline" / "metrics.json"),
        "F_pipeline": load(
            RES / "ch-13-inference/runs/val_full/combo_flip_tile_nms05/metrics.json"
        ),
    }


LABELS = {
    "C_single": "CH-14 final_rtdetr, single pass",
    "C_pipeline": "CH-14 final_rtdetr, CH-13 pipeline",
    "F_single": "fallback rtdetr_baseline, single pass",
    "F_pipeline": "fallback rtdetr_baseline, CH-13 pipeline",
}


def epochs_table() -> list[dict]:
    rows = []
    times = {}
    with open(
        RES / "final_rtdetr/ultralytics/results.csv", newline="", encoding="utf-8"
    ) as f:
        for r in csv.DictReader(f):
            times[int(r["epoch"])] = {k.strip(): v for k, v in r.items()}
    base = {e["epoch"]: e for e in load(RES / "rtdetr_baseline/curve.json")}
    previous = 0.0
    for e in load(RES / "final_rtdetr/curve.json"):
        t = times.get(e["epoch"], {})
        elapsed = float(t["time"]) if t.get("time") else None
        rows.append(
            {
                "epoch": e["epoch"],
                "checkpoint": e["checkpoint"],
                "images_seen": e["epoch"] * 6072,
                "train_giou_loss": t.get("train/giou_loss"),
                "train_cls_loss": t.get("train/cls_loss"),
                "train_l1_loss": t.get("train/l1_loss"),
                "ultralytics_elapsed_s": elapsed,
                "epoch_seconds": round(elapsed - previous, 1) if elapsed else None,
                **{f"val_fast_{k}": round(v, 4) for k, v in e["val_fast"].items()},
                "rtdetr_baseline_val_fast_mAP": round(
                    base[e["epoch"]]["val_fast"]["mAP"], 4
                )
                if e["epoch"] in base
                else None,
            }
        )
        previous = elapsed or previous
    return rows


def decide(m: dict) -> dict:
    """Rule B of selection_rule.md, applied to C_pipeline vs F_pipeline."""
    c, f = m["C_pipeline"], m["F_pipeline"]
    checks = []
    for k in SECONDARY:
        checks.append(
            {
                "check": k,
                "C": c["metrics"][k],
                "F": f["metrics"][k],
                "C_leads": c["metrics"][k] > f["metrics"][k],
            }
        )
    for cb, fb in zip(c["density"], f["density"]):
        assert cb["bin"] == fb["bin"]
        checks.append(
            {
                "check": f"density {cb['bin']}",
                "C": cb["mAP"],
                "F": fb["mAP"],
                "C_leads": cb["mAP"] > fb["mAP"],
            }
        )
    lead = c["metrics"]["mAP"] - f["metrics"]["mAP"]
    run = load(HERE / "final_rtdetr/val_full/run.json")
    config = load(RES / "final_rtdetr/config.json")
    identity = {
        "train_split": config["train_split"],
        "num_images": config["export_manifest"]["num_images"],
        "epochs_completed": len(load(RES / "final_rtdetr/curve.json")),
        "train_args_epochs": run["train_args_epochs"],
        "train_args_fraction": run["train_args_fraction"],
        "checkpoint_sha256": run["checkpoint_sha256"],
    }
    rule = {
        "B1_mAP_lead_ge_spread": lead >= SPREAD,
        "B2_secondary_wins_ge_5_of_9": sum(x["C_leads"] for x in checks) >= 5,
        "B3_AP_small_guard": c["metrics"]["AP_small"]
        >= f["metrics"]["AP_small"] - APS_GUARD,
        "B4_identity": identity["train_split"] == "train_final"
        and identity["num_images"] == 6072
        and identity["epochs_completed"] == 6
        and identity["train_args_epochs"] == 6
        and identity["train_args_fraction"] in (None, 1.0),
    }
    winner = "C" if all(rule.values()) else "F"
    return {
        "rule": "results/ch-14-final/selection_rule.md (rule B)",
        "compared": "val_full, CH-13 pipeline configs/runs/final_inference.json, project evaluator maxDets 500",
        "C": "CH-14 final_rtdetr last.pt (epoch 6)",
        "F": "CH-11 rtdetr_baseline last.pt (sha256 da5d3625...e872)",
        "mAP": {"C": c["metrics"]["mAP"], "F": f["metrics"]["mAP"], "lead": lead},
        "spread": SPREAD,
        "secondary_checks": checks,
        "secondary_wins": sum(x["C_leads"] for x in checks),
        "identity": identity,
        "criteria": rule,
        "tie": 0 < lead < SPREAD,
        "primary_submission": "CH-14 final_rtdetr"
        if winner == "C"
        else "fallback rtdetr_baseline + CH-13 pipeline",
        "single_pass_data_strategy_gain": m["C_single"]["metrics"]["mAP"]
        - m["F_single"]["metrics"]["mAP"],
    }


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def paired_bars(ax, labels, c_vals, f_vals) -> None:
    import numpy as np

    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2 - 0.01, c_vals, width, color=BLUE, label=LABELS["C_pipeline"])
    ax.bar(
        x + width / 2 + 0.01, f_vals, width, color=ORANGE, label=LABELS["F_pipeline"]
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(max(c_vals), max(f_vals)) * 1.25)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left", ncol=2)
    style(ax)


def figures(m: dict, epochs: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"}
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    base = load(RES / "rtdetr_baseline/curve.json")
    for series, color, label in (
        (
            [(r["epoch"], r["val_fast_mAP"]) for r in epochs],
            BLUE,
            "final_rtdetr (train_final, 6,072 frames)",
        ),
        (
            [(e["epoch"], e["val_fast"]["mAP"]) for e in base],
            ORANGE,
            "rtdetr_baseline (train_rubric, 2,800 frames)",
        ),
    ):
        xs, ys = zip(*series)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8, label=label)
        ax.annotate(f"{ys[-1]:.4f}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK)  # fmt: skip
    ax.set_xlabel("epoch")
    ax.set_ylabel("val_fast mAP@[.5:.95]")
    ax.set_title(
        "Per-epoch val_fast mAP (project evaluator, single pass)", color=INK, loc="left"
    )
    ax.set_xticks(range(1, 7))
    ax.legend(frameon=False, labelcolor=INK, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "curve.png", dpi=130)
    plt.close(fig)

    c, f = m["C_pipeline"], m["F_pipeline"]
    names = [r["name"] for r in c["per_class"]]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    paired_bars(
        ax, names, [r["AP"] for r in c["per_class"]], [r["AP"] for r in f["per_class"]]
    )
    ax.set_ylabel("val_full AP@[.5:.95]")
    ax.set_title("Per-class AP, CH-13 pipeline, val_full", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "per_class.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bins = [f"{b['bin']}\n({b['num_images']} frames)" for b in c["density"]]
    paired_bars(
        ax, bins, [b["mAP"] for b in c["density"]], [b["mAP"] for b in f["density"]]
    )
    ax.set_xlabel("target objects per frame")
    ax.set_ylabel("val_full mAP@[.5:.95]")
    ax.set_title(
        "mAP by scene density, CH-13 pipeline, val_full", color=INK, loc="left"
    )
    fig.tight_layout()
    fig.savefig(out / "density.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hash-checkpoints", action="store_true")
    args = parser.parse_args()

    m = models()
    epochs = epochs_table()
    write_csv(HERE / "epochs.csv", epochs)
    write_csv(
        HERE / "comparison.csv",
        [
            {
                "model": key,
                "label": LABELS[key],
                **{k: round(v, 4) for k, v in r["metrics"].items()},
                **{f"density_{b['bin']}": round(b["mAP"], 4) for b in r["density"]},
                "num_predictions": r["num_predictions"],
                "images_over_500": r["images_over_headline_max_dets"],
            }
            for key, r in m.items()
        ],
    )
    write_csv(
        HERE / "per_class.csv",
        [
            {
                "model": key,
                **{
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in row.items()
                },
            }
            for key, r in m.items()
            for row in r["per_class"]
        ],
    )
    write_csv(
        HERE / "density.csv",
        [{"model": key, **b} for key, r in m.items() for b in r["density"]],
    )
    decision = decide(m)
    (HERE / "selection.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    if args.hash_checkpoints:
        weights = REPO / "outputs/final_rtdetr/train/weights"
        (HERE / "checkpoints.json").write_text(
            json.dumps(
                {
                    "location": "outputs/final_rtdetr/train/weights (git-ignored)",
                    "files": {
                        p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                        for p in sorted(weights.glob("*.pt"))
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    figures(m, epochs)
    print(
        json.dumps(
            {
                k: decision[k]
                for k in ("mAP", "secondary_wins", "criteria", "primary_submission")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
