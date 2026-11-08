"""Regenerate CH-08 tables and the validation learning curves from run JSON."""

import csv
import json
import math
from pathlib import Path

from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
RETINA = ("retina_baseline", "retina_anchors", "retina_focal_gamma15")
FRCNN = ("frcnn_baseline", "frcnn_rpn_pre3000", "frcnn_roi_pos50")


def read(run, artifact):
    return json.loads(
        (HERE.parent / run / f"{artifact}.json").read_text(encoding="utf-8")
    )


def write(name, rows):
    with (HERE / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    comparison, classes, density = [], [], []
    figure = Figure(figsize=(8, 4), layout="constrained")
    axis = figure.subplots()
    for run in (*RETINA, *FRCNN):
        curve = read(run, "curve")
        runtime = read(run, "runtime")
        full = read(run, "metrics")["val_full"]
        speed = read(run, "speed")["inference"]
        assert [row["global_step"] for row in curve] == [1400 * i for i in range(1, 7)]
        assert all(
            row["images_seen"] == 2800 and math.isfinite(row["loss"]) for row in curve
        )
        assert runtime["images_seen"] == 16800
        assert full["num_images"] == 2846 and full["max_dets"] == 500
        assert full["images_over_headline_max_dets"] == 0
        assert (
            speed["batch_size"] == 1
            and speed["timed_batches"] == 30
            and speed["fps"] > 0
        )
        comparison.append(
            {
                "run": run,
                **{f"val_fast_{k}": v for k, v in curve[-1]["val_fast"].items()},
                **{f"val_full_{k}": v for k, v in full["metrics"].items()},
                "fps": speed["fps"],
                "seconds": runtime["this_process_seconds"],
            }
        )
        classes.extend(
            {
                "run": run,
                "class": row["name"],
                "targets": row["num_gt"],
                "AP": row["AP"],
            }
            for row in full["per_class"]
        )
        density.extend({"run": run, **row} for row in full["density"])
        if run in RETINA or run == "frcnn_rpn_pre3000":
            axis.plot(
                [row["epoch"] for row in curve],
                [row["val_fast"]["mAP"] for row in curve],
                marker="o",
                label=run,
                linestyle="--" if run in FRCNN else "-",
            )
    selected = max(comparison[: len(RETINA)], key=lambda row: row["val_fast_mAP"])[
        "run"
    ]
    for row in comparison:
        row["selected_retina"] = row["run"] == selected
    write("comparison.csv", comparison)
    write("per_class.csv", classes)
    write("density.csv", density)
    axis.set(
        xlabel="Epoch",
        ylabel="val_fast mAP",
        title="Equal-budget RetinaNet variants and selected Faster R-CNN",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    (HERE / "figures").mkdir(exist_ok=True)
    figure.savefig(HERE / "figures" / "val_fast_curves.png", dpi=150)
    print(f"Selected RetinaNet by final val_fast mAP: {selected}")


if __name__ == "__main__":
    main()
