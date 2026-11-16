"""OPT-12 reference R: the Ultralytics validator on the start weights, with the in-training validator's settings.

In training (val true) Ultralytics validates the EMA in FP16 at batch 2 on the data.yaml `val` list (val_fast,
unmodified copy), conf 0.001, IoU 0.7, max_det 500. R is the same measurement on soup_w075, used only for the frozen
catastrophe check and as the trend baseline. Never a headline number (those come from movd/evaluate.py).

    python scripts/opt12_refval.py --weights ../MOVD-opt09/outputs/opt09/soup/soup_w075.pt --device 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movd import opt12, rtdetr  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--data", default="outputs/yolo_data_final/data.yaml")
    ap.add_argument("--out", type=Path, default=opt12.EVID / "reference_R.json")
    a = ap.parse_args()
    t0 = time.perf_counter()
    model = rtdetr.load_model(a.weights, "rtdetr")
    m = model.val(data=str(opt12.REPO / a.data), imgsz=1344, batch=2, half=True, conf=0.001, iou=0.7, max_det=500,
                  device=a.device, workers=2, plots=False, verbose=False, project=str(opt12.OUT / "refval"),
                  name="w075", exist_ok=True)
    rec = {"weights": a.weights.resolve().as_posix(), "weights_sha256": opt12.sha256(a.weights),
           "monitor_mAP50_95": float(m.box.map), "monitor_mAP50": float(m.box.map50),
           "settings": {"imgsz": 1344, "batch": 2, "half": True, "conf": 0.001, "iou": 0.7, "max_det": 500,
                        "split": "val_fast list of data.yaml"},
           "seconds": round(time.perf_counter() - t0, 1), "at_kyiv": opt12.stamp(),
           "note": "Ultralytics validator, monitoring only; same settings as the in-training validator"}
    opt12.save_json(a.out, rec)
    print(rec)


if __name__ == "__main__":
    main()
