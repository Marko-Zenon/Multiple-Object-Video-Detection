"""Evaluate Ultralytics checkpoints with the project evaluator (CH-03). Never uses Ultralytics' metrics.

Subcommands (all take a required --device, '0' or 'cpu'):

* `curve --run-id R`: every `outputs/R/train/weights/epoch<N>.pt` -> val_fast predictions -> movd.evaluate.
  Resumable: epochs already in results/R/curve.json are skipped. Writes results/R/curve.json and
  results/R/figures/curve.png. Epoch = N + 1 (Ultralytics numbers checkpoints from 0).
* `final --run-id R`: the final checkpoint (`last.pt`) on val_full -> results/R/metrics.json, per_class.csv,
  speed.json. Raw predictions go to the ignored outputs/R/val_full_predictions.json.
* `predict --weights W --split S --out F`: COCO-results JSON for any checkpoint and val split.

Prediction settings: conf 0.001, NMS IoU 0.7, max_det 500, imgsz from the run config (protocol 1344), original
val images, boxes in original pixels, then the shared 500-per-image cap.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import data, ultra  # noqa: E402
from movd import evaluate as ev  # noqa: E402


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def run_settings(run_id: str) -> dict:
    config = json.loads(
        (data.REPO_ROOT / "results" / run_id / "config.json").read_text(
            encoding="utf-8"
        )
    )
    return config["ultralytics_train_kwargs"]


def predict(
    weights: Path, split: str, device: str, imgsz: int, half: bool, root
) -> tuple[list[dict], float]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    start = time.perf_counter()
    detections = ultra.predict_coco(
        model, ultra.val_frames(split, root), imgsz=imgsz, device=device, half=half
    )
    return detections, time.perf_counter() - start


def curve(args) -> None:
    settings = run_settings(args.run_id)
    results = data.REPO_ROOT / "results" / args.run_id
    weights_dir = data.REPO_ROOT / "outputs" / args.run_id / "train" / "weights"
    path = results / "curve.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    done = {row["epoch"] for row in rows}
    checkpoints = sorted(weights_dir.glob("epoch*.pt"), key=lambda p: int(p.stem[5:]))
    if not checkpoints:
        raise FileNotFoundError(f"no epoch checkpoints in {weights_dir}")
    for checkpoint in checkpoints:
        epoch = int(checkpoint.stem[5:]) + 1
        if epoch in done:
            continue
        detections, seconds = predict(
            checkpoint,
            "val_fast",
            args.device,
            settings["imgsz"],
            args.half,
            args.dataset_root,
        )
        save_json(
            data.REPO_ROOT
            / "outputs"
            / args.run_id
            / "val_fast_predictions"
            / f"epoch{epoch - 1}.json",
            detections,
        )
        result = ev.evaluate_split(detections, "val_fast", args.dataset_root)
        rows.append(
            {
                "epoch": epoch,
                "checkpoint": checkpoint.name,
                "val_fast": result["metrics"],
                f"val_fast_{ev.LEGACY_KEY}": result[f"metrics_{ev.LEGACY_KEY}"],
                "num_predictions": result["num_predictions"],
                "predict_seconds": round(seconds, 1),
            }
        )
        rows.sort(key=lambda r: r["epoch"])
        save_json(path, rows)
        print(
            f"epoch {epoch}: val_fast mAP {result['metrics']['mAP']:.4f} AP50 {result['metrics']['AP50']:.4f}"
        )
    plot_curve(rows, results / "figures" / "curve.png", args.run_id)


def plot_curve(rows, path: Path, run_id: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ("mAP", "AP50", "AP_small"):
        ax.plot(epochs, [r["val_fast"][key] for r in rows], marker="o", label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val_fast (movd.evaluate, maxDets 500)")
    ax.set_title(run_id)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def speed(weights: Path, device: str, imgsz: int, half: bool, root) -> dict:
    import cv2
    from ultralytics import YOLO

    from movd.speed import benchmark

    model = YOLO(str(weights))
    frames = [cv2.imread(str(p)) for _, p in ultra.val_frames("val_full", root)[:30]]
    kwargs = ultra.predict_kwargs(imgsz, device, half)
    return benchmark(
        lambda frame: model.predict(frame, **kwargs),
        frames,
        device="cuda" if device != "cpu" else "cpu",
        batch_size=1,
        warmup=10,
        input_size=f"imgsz {imgsz} (letterbox, long side)",
        note="First 30 val_full frames preloaded as BGR arrays (no disk I/O); includes Ultralytics letterbox, normalisation, NMS and box rescaling; conf 0.001, max_det 500",
    )


def final(args) -> None:
    settings = run_settings(args.run_id)
    results = data.REPO_ROOT / "results" / args.run_id
    output = data.REPO_ROOT / "outputs" / args.run_id
    weights = args.weights or output / "train" / "weights" / "last.pt"
    detections, seconds = predict(
        weights,
        "val_full",
        args.device,
        settings["imgsz"],
        args.half,
        args.dataset_root,
    )
    save_json(output / "val_full_predictions.json", detections)
    result = ev.evaluate_split(detections, "val_full", args.dataset_root)
    result["checkpoint"] = str(Path(weights).relative_to(data.REPO_ROOT).as_posix())
    result["predict_seconds"] = round(seconds, 1)
    ev.write_outputs(result, results)
    print(ev.format_summary(result))
    save_json(
        results / "speed.json",
        speed(weights, args.device, settings["imgsz"], args.half, args.dataset_root),
    )


def predict_cmd(args) -> None:
    detections, seconds = predict(
        args.weights, args.split, args.device, args.imgsz, args.half, args.dataset_root
    )
    save_json(args.out, detections)
    print(f"{len(detections)} detections in {seconds:.1f} s -> {args.out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("curve", "final", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--device", required=True)
        p.add_argument("--dataset-root", type=Path, default=None)
        p.add_argument("--half", action="store_true", help="FP16 inference (GPU only)")
        if name in ("curve", "final"):
            p.add_argument("--run-id", required=True)
        if name == "final":
            p.add_argument("--weights", type=Path, default=None)
        if name == "predict":
            p.add_argument("--weights", type=Path, required=True)
            p.add_argument("--split", choices=("val_fast", "val_full"), required=True)
            p.add_argument("--out", type=Path, required=True)
            p.add_argument(
                "--imgsz",
                type=int,
                default=data.load_protocol()["rubric_budget"]["input_long_side"],
            )
    args = parser.parse_args()
    {"curve": curve, "final": final, "predict": predict_cmd}[args.command](args)


if __name__ == "__main__":
    main()
