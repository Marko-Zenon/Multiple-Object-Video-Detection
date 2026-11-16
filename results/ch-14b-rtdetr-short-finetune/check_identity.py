"""CH-14b identity checks of a checkpoint before it is evaluated (CPU only), see selection_rule.md (A0 / R4).

    python results/ch-14b-rtdetr-short-finetune/check_identity.py stageA --weights <CH-14 epoch0.pt>
    python results/ch-14b-rtdetr-short-finetune/check_identity.py stageB --weights <fine-tune epoch checkpoint>

Writes results/ch-14b-rtdetr-short-finetune/<stage>_identity.json and exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from movd import data, rtdetr  # noqa: E402

STAGE_A_SHA = "6ff4a430188405b5d13356872c63f141c4d3dbc7b747983cdf733ce386aa5b0b"
FALLBACK_SHA = "da5d36258918e7fdddcb9cfe6e87f6a78ef8ee1a04e65deb3c0eedb42fb6e872"
NAMES = data.load_protocol()["label_map"][
    "names"
]  # person, bicycle, car, van, truck, tricycle, bus, motor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(path: Path) -> str:
    """sha256 of a text file with LF line endings (the committed blob; Windows checkouts may use CRLF)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def architecture(weights: Path) -> dict:
    """Head type, queries, classes and layer count of a checkpoint vs the official rtdetr-l.pt."""
    model = rtdetr.load_model(weights, "rtdetr")
    reference = rtdetr.load_model(REPO / "outputs/weights/rtdetr-l.pt", "rtdetr")
    names = (
        model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    )
    return {
        "head": type(rtdetr.decoder(model)).__name__,
        "num_queries": rtdetr.num_queries(model),
        "nc": int(rtdetr.decoder(model).nc),
        "names": [names[i] for i in sorted(names)],
        "layers": len(model.model.model),
        "reference_layers": len(reference.model.model),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stageA", "stageB"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--run-id", default="ch14b_finetune", help="stageB: the run directory name"
    )
    args = parser.parse_args()

    import torch

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    train_args = ckpt.get("train_args", {})
    arch = architecture(args.weights)
    split_sha = text_sha256(data.SPLITS_DIR / "train_final.json")
    run = "final_rtdetr" if args.stage == "stageA" else args.run_id
    record = json.loads(
        (REPO / "results" / run / "config.json").read_text(encoding="utf-8")
    )
    info = {
        "weights": args.weights.resolve().as_posix(),
        "sha256": sha256(args.weights),
        "bytes": args.weights.stat().st_size,
        "epoch_field": ckpt.get("epoch"),
        "train_args": {k: train_args.get(k) for k in ("epochs", "imgsz", "batch", "seed", "fraction", "data",
                                                         "optimizer", "lr0", "momentum", "warmup_bias_lr", "amp")},
        "architecture": arch,
        "run_record": f"results/{run}/config.json",
        "run_train_split": record["train_split"],
        "run_manifest_num_images": record["export_manifest"]["num_images"],
        "run_split_file_sha256": record["train_split_file_sha256"],
        "train_final_split_file_sha256_now": split_sha,
    }  # fmt: skip
    del ckpt
    checks = {
        "architecture_rtdetr_l": arch["head"] == "RTDETRDecoder"
        and arch["num_queries"] == 300
        and arch["nc"] == 8
        and arch["names"] == NAMES
        and arch["layers"] == arch["reference_layers"],
        "imgsz_1344_batch_1_seed_42": info["train_args"]["imgsz"] == 1344
        and info["train_args"]["batch"] == 1
        and info["train_args"]["seed"] == 42,
        "not_smoke": info["train_args"]["fraction"] in (None, 1.0),
        "data_is_train_final_export": "yolo_data_final"
        in str(info["train_args"]["data"]),
        "run_on_train_final_6072": info["run_train_split"] == "train_final"
        and info["run_manifest_num_images"] == 6072,
        "split_file_unchanged": info["run_split_file_sha256"] == split_sha,
    }
    if args.stage == "stageA":
        checks["sha256_matches_ch14_checkpoints_json"] = info["sha256"] == STAGE_A_SHA
        checks["epoch1_of_6"] = (
            info["epoch_field"] == 0 and info["train_args"]["epochs"] == 6
        )
    else:
        checks["init_is_fallback"] = record["weights_sha256"] == FALLBACK_SHA
        checks["epochs_2_lr0"] = (
            info["train_args"]["epochs"] == 2
            and info["train_args"]["optimizer"] == "AdamW"
            and abs(info["train_args"]["lr0"] - 0.0000833) < 1e-12
        )
        info["init_weights_sha256"] = record["weights_sha256"]
    info["checks"] = checks
    info["passed"] = all(checks.values())
    out = HERE / f"{args.stage}_identity.json"
    out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2), "\npassed:", info["passed"])
    sys.exit(0 if info["passed"] else 1)


if __name__ == "__main__":
    main()
