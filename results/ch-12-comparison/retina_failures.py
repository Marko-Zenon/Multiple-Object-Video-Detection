"""CH-12: failure cases for RetinaNet (CH-08 stored none), plus a checkpoint-identity check (GPU inference only, no training).

    python results/ch-12-comparison/retina_failures.py

Loads the CH-08-selected checkpoint (outputs/ch08/outputs/retina_anchors/epoch_0006.pt in the main checkout, read-only). It
predicts val_full with the CH-04 code path (`movd.tv_train.predict_coco`, the settings stored in results/retina_anchors/config.json)
and re-evaluates with movd.evaluate. The mAP must reproduce results/retina_anchors/metrics.json; that check proves the checkpoint
is the one that produced the stored numbers. Then it applies the CH-07 failure rule (`movd.ultra.find_failure_cases`) and draws the
cases with scripts/failures_ultra.py's `draw_case`. Raw predictions go to the git-ignored outputs/retina_anchors/. Tracked outputs:
results/ch-12-comparison/retina_anchors_failures/{identity.json, failure_cases.csv, figures/*.png}.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from movd import data, tv_train, ultra  # noqa: E402
from movd import evaluate as ev  # noqa: E402

RUN = "retina_anchors"
CHECKPOINT = (
    ROOT.parent
    / "Multiple-Object-Video-Detection"
    / "outputs"
    / "ch08"
    / "outputs"
    / RUN
    / "epoch_0006.pt"
)
OUT = ROOT / "results" / "ch-12-comparison" / f"{RUN}_failures"


def main() -> None:
    import torch

    config = json.loads(
        (ROOT / "results" / RUN / "config.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = tv_train.build_model(config, initialize_weights=False)
    model.load_state_dict(checkpoint["model"])
    model.cuda().eval()
    frames = tv_train.validation_frames(config, "val_full")
    with torch.no_grad():
        predictions = tv_train.predict_coco(
            model, tv_train.make_loader(frames, config), config["family"], config["amp"]
        )
    raw = ROOT / "outputs" / RUN / "val_full_predictions.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(predictions), encoding="utf-8")

    result = ev.evaluate_split(predictions, "val_full")
    stored = json.loads(
        (ROOT / "results" / RUN / "metrics.json").read_text(encoding="utf-8")
    )["val_full"]["metrics"]
    diffs = {k: abs(result["metrics"][k] - stored[k]) for k in stored}
    OUT.mkdir(parents=True, exist_ok=True)
    identity = {
        "checkpoint": CHECKPOINT.relative_to(ROOT.parent).as_posix(),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        # results/<run>/config.json adds provenance keys (weights, seed, protocol_version, ...) to the checkpoint's config
        "config_shared_keys_match_checkpoint": all(
            config[k] == v for k, v in checkpoint["config"].items()
        ),
        "config_keys_added_in_results": sorted(set(config) - set(checkpoint["config"])),
        "reproduced_val_full": result["metrics"],
        "stored_val_full": stored,
        "max_abs_diff": max(diffs.values()),
        "reproduces_stored_metrics": max(diffs.values()) < 1e-3,
        "num_predictions": result["num_predictions"],
        "note": "AMP inference on GPU; small differences (< 1e-3) come from non-deterministic CUDA kernels",
    }
    (OUT / "identity.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8"
    )

    spec = importlib.util.spec_from_file_location(
        "failures_ultra", ROOT / "scripts" / "failures_ultra.py"
    )
    failures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(failures)
    coco = data.load_coco("val")
    images = {img["id"]: img for img in coco["images"]}
    entries = data.split_annotations(coco)
    ordered = [images[i] for i in data.split_image_ids("val_full")]
    cases = ultra.find_failure_cases(ordered, entries, predictions)
    rows = [
        failures.draw_case(
            kind,
            case,
            entries[case["image"]["id"]],
            OUT / "figures" / f"{kind}.png",
            None,
        )
        for kind, case in cases.items()
    ]
    with (OUT / "failure_cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                k: identity[k]
                for k in (
                    "max_abs_diff",
                    "reproduces_stored_metrics",
                    "config_shared_keys_match_checkpoint",
                )
            }
        )
    )
    for row in rows:
        print(f"{row['case']}: {row['frame']} - {row['observation']}")


if __name__ == "__main__":
    main()
