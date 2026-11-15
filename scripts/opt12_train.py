"""OPT-12 training launcher: the scripts/train_ultra.py path plus four OPT-12 additions.

1. Frozen BatchNorm statistics: every nn.BatchNorm2d is put in eval mode before every batch (on_train_batch_start, the
   first hook after the trainer's per-epoch model.train()). Affine parameters stay trainable; the decoder head stays in
   train mode. A hash over all BN running buffers is recorded at the start and after every epoch.
2. Start-weight identity: at train start every tensor of the start checkpoint's EMA must equal the trainer's model
   tensor (cast to float32), so the run provably starts from soup_w075 (or the recorded fallback).
3. Augmentation proof: the training dataset's transform pipeline is recorded, and in a smoke run a grid of augmented
   training samples is saved as a figure.
4. Clock and catastrophe stops, raised from on_fit_epoch_end (after the epoch's checkpoints are written): stop when
   another epoch could not finish by --stop-at, and abort when the monitor after epoch 2 is below --reference - 0.060.

Resume: only from a checkpoint that still holds an optimizer and an epoch (last.pt, else the highest epoch<N>.pt),
because Ultralytics silently starts a new default training from a stripped file.

    python scripts/opt12_train.py --config configs/runs/opt12_reg_ft.json --device 0 --stop-at "2026-11-15 06:55" \
        --reference 0.30
    python scripts/opt12_train.py --config ... --device 0 --resume --stop-at ... --reference ...

The Ultralytics validator (val true, val_fast list) is a monitoring trend only; headline numbers come from
movd/evaluate.py. Ultralytics is AGPL-3.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from movd import data, opt12, rtdetr, ultra  # noqa: E402


class StopRun(Exception):
    """Intentional stop after an epoch's checkpoints are written (clock, catastrophe or smoke stop)."""

    def __init__(self, status: str):
        super().__init__(status)
        self.status = status


def bn_modules(model):
    from torch import nn
    from ultralytics.utils.torch_utils import unwrap_model

    return [m for m in unwrap_model(model).modules() if isinstance(m, nn.BatchNorm2d)]


def bn_hash(model) -> str:
    h = hashlib.sha256()
    for m in bn_modules(model):
        for buf in (m.running_mean, m.running_var):
            h.update(buf.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def resumable(path: Path) -> bool:
    import torch

    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - a corrupt file is simply not resumable
        return False
    return ck.get("optimizer") is not None and int(ck.get("epoch", -1)) >= 0


def resume_source(weights_dir: Path) -> Path | None:
    last = weights_dir / "last.pt"
    if last.exists() and resumable(last):
        return last
    epochs = sorted(weights_dir.glob("epoch*.pt"), key=lambda p: int(p.stem[5:]), reverse=True)
    for p in epochs:
        if resumable(p):
            return p
    return None


def start_identity(trainer, start_ckpt: Path) -> dict:
    import torch
    from ultralytics.utils.torch_utils import unwrap_model

    ref = torch.load(start_ckpt, map_location="cpu", weights_only=False)["ema"].state_dict()
    cur = unwrap_model(trainer.model).state_dict()
    missing = [k for k in ref if k not in cur]
    diffs = []
    for k, v in ref.items():
        if k in cur and v.is_floating_point():
            d = (cur[k].detach().float().cpu() - v.float()).abs().max().item() if v.numel() else 0.0
            if d != 0.0:
                diffs.append((k, d))
    return {"start_checkpoint": start_ckpt.as_posix(), "tensors": len(ref), "missing_in_model": missing[:20],
            "num_missing": len(missing), "num_nonzero_diff": len(diffs), "max_diff": max((d for _, d in diffs), default=0.0),
            "first_diffs": diffs[:10], "exact": not missing and not diffs}


def augmentation_record(trainer, figure: Path | None) -> dict:
    ds = trainer.train_loader.dataset
    rec = {"transforms": repr(ds.transforms), "hyp": {k: getattr(trainer.args, k) for k in (
        "mosaic", "mixup", "cutmix", "copy_paste", "scale", "translate", "degrees", "shear", "perspective",
        "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "close_mosaic")}}
    if figure is not None:
        import cv2
        import numpy as np

        tiles, shapes = [], []
        for i in range(0, min(len(ds), 64), 8):
            item = ds[i]
            img = item["img"].permute(1, 2, 0).cpu().numpy().copy()  # CHW RGB uint8 -> HWC
            h, w = img.shape[:2]
            boxes = item["bboxes"].cpu().numpy()  # normalised xywh
            for cx, cy, bw, bh in boxes:
                p0 = (int((cx - bw / 2) * w), int((cy - bh / 2) * h))
                p1 = (int((cx + bw / 2) * w), int((cy + bh / 2) * h))
                cv2.rectangle(img, p0, p1, (255, 0, 0), 2)
            shapes.append([h, w, int(len(boxes))])
            tiles.append(cv2.resize(img, (448, 448)))
        rows = [np.concatenate(tiles[r:r + 4], axis=1) for r in range(0, len(tiles), 4) if len(tiles[r:r + 4]) == 4]
        if rows:
            figure.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(figure), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
        rec["sample_shapes_hw_boxes"] = shapes
        rec["figure"] = figure.as_posix()
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-at", help="Kyiv 'YYYY-MM-DD HH:MM': stop after the epoch if the next cannot finish")
    parser.add_argument("--reference", type=float, help="Ultralytics monitor of the start weights (catastrophe check)")
    parser.add_argument("--smoke-stop-after-epoch", type=int, help="Smoke only: stop after this many epochs")
    parser.add_argument("--no-bn-eval", action="store_true", help="Diagnostics only; the OPT-12 run uses BN eval")
    args = parser.parse_args()

    config = ultra.load_run_config(args.config)
    if args.run_id:
        config = ultra.validate_run_config({**config, "run_id": args.run_id})
    if args.smoke_stop_after_epoch is not None and "smoke_fraction" not in config:
        parser.error("--smoke-stop-after-epoch needs a smoke config")
    run_id = config["run_id"]
    output = data.REPO_ROOT / "outputs" / run_id
    results = data.REPO_ROOT / "results" / run_id
    weights_dir = output / "train" / "weights"
    data_yaml = data.REPO_ROOT / config["data_yaml"]
    manifest = ultra.train_manifest(config, data_yaml)
    start_ckpt = (data.REPO_ROOT / config["weights"]).resolve()
    stop_at = (datetime.fromisoformat(args.stop_at).replace(tzinfo=opt12.KYIV) if args.stop_at else None)

    import torch

    if args.resume:
        source = resume_source(weights_dir)
        if source is None:
            raise SystemExit(f"no resumable checkpoint (optimizer + epoch) in {weights_dir}")
    else:
        if results.exists() or output.exists():
            raise SystemExit(f"{results} or {output} exists; use --resume or a new --run-id")
        output.mkdir(parents=True)
        results.mkdir(parents=True, exist_ok=True)
        kwargs = ultra.train_kwargs(config, data_yaml, output, args.device)
        import train_ultra as tu

        opt12.save_json(results / "config.json", {
            "run_config": config, "ultralytics_train_kwargs": kwargs, "weights": config["weights"],
            "weights_sha256": opt12.sha256(start_ckpt), "protocol_version": data.load_protocol()["version"],
            "train_split": config["train_split"], "export_manifest": manifest,
            "bn_eval": not args.no_bn_eval, "stop_at": args.stop_at, "reference": args.reference,
            "headline_evaluator": "movd/evaluate.py (Ultralytics validator = monitoring trend only)"})
        opt12.save_json(results / "hardware.json", tu.hardware(args.device))

    obs_path = results / "observations.json"
    obs = opt12.load_json(obs_path) if obs_path.exists() else {"processes": [], "epochs": []}
    proc = {"resume": args.resume, "started": opt12.stamp(), "batches": 0}
    obs["processes"].append(proc)
    smoke = "smoke_fraction" in config
    t0 = time.perf_counter()
    state = {"epoch_start": None}

    model = rtdetr.load_model(source if args.resume else start_ckpt, "rtdetr")

    cache = {}

    def on_train_start(trainer):
        proc.update(start_epoch=trainer.start_epoch, samples=len(trainer.train_loader.dataset),
                    batches_per_epoch=len(trainer.train_loader), epochs=trainer.epochs, amp=bool(trainer.amp),
                    accumulate=trainer.accumulate, optimizer=type(trainer.optimizer).__name__,
                    lr0=trainer.args.lr0, bn_hash_start=bn_hash(trainer.model))
        frozen = [n for n, p in trainer.model.named_parameters() if not p.requires_grad]
        proc["frozen_params"] = len(frozen)
        proc["frozen_prefixes"] = sorted({".".join(n.split(".")[:2]) for n in frozen})
        if not args.resume:
            ident = start_identity(trainer, start_ckpt)
            opt12.save_json(results / "start_identity.json", ident)
            proc["start_identity_exact"] = ident["exact"]
            if not ident["exact"]:
                raise SystemExit(f"start weights not loaded exactly: {ident}")
        fig = results / "figures" / "augmented_samples.jpg" if smoke and not args.resume else None
        opt12.save_json(results / ("augmentation_resume.json" if args.resume else "augmentation.json"),
                        augmentation_record(trainer, fig))
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)
        state["epoch_start"] = time.perf_counter()
        opt12.save_json(obs_path, obs)

    def on_train_epoch_start(trainer):
        state["epoch_start"] = time.perf_counter()

    def on_train_batch_start(trainer):
        if args.no_bn_eval:
            return
        if cache.get("key") != id(trainer.model):
            cache["key"] = id(trainer.model)
            cache["bns"] = bn_modules(trainer.model)
        for m in cache["bns"]:
            m.eval()

    def on_train_batch_end(trainer):
        if not torch.isfinite(trainer.loss).all():
            raise FloatingPointError("non-finite training loss")
        proc["batches"] += 1
        el = round(time.perf_counter() - t0, 3)
        proc.setdefault("first_batch_elapsed_seconds", el)
        proc["last_batch_elapsed_seconds"] = el
        if proc["batches"] in (200, 1000):
            proc[f"batches_{proc['batches']}_elapsed"] = el

    def on_fit_epoch_end(trainer):
        ep = trainer.epoch + 1  # 1-based
        secs = time.perf_counter() - (state["epoch_start"] or t0)
        monitor = float(trainer.metrics.get("metrics/mAP50-95(B)", float("nan")))
        rec = {"epoch": ep, "t": opt12.stamp(), "epoch_seconds_incl_val": round(secs, 1), "monitor_mAP50_95": monitor,
               "monitor_mAP50": float(trainer.metrics.get("metrics/mAP50(B)", float("nan"))),
               "bn_hash": bn_hash(trainer.model), "bn_hash_equal_start": bn_hash(trainer.model) == proc["bn_hash_start"],
               "lr_pg0": trainer.optimizer.param_groups[0]["lr"],
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(trainer.device) if trainer.device.type == "cuda" else 0,
               "checkpoint": (weights_dir / f"epoch{trainer.epoch}.pt").as_posix(),
               "checkpoint_exists": (weights_dir / f"epoch{trainer.epoch}.pt").exists()}
        obs["epochs"].append(rec)
        opt12.save_json(obs_path, obs)
        opt12.event("epoch_end", run_id=run_id, **rec)
        if args.reference is not None and opt12.catastrophe(ep, monitor, args.reference):
            raise StopRun(f"catastrophe_abort_after_epoch_{ep}")
        if args.smoke_stop_after_epoch is not None and ep >= args.smoke_stop_after_epoch and ep < trainer.epochs:
            raise StopRun(f"smoke_stop_after_epoch_{ep}")
        if stop_at is not None and ep < trainer.epochs and opt12.clock_stop(opt12.now(), secs, stop_at):
            raise StopRun(f"clock_stop_after_epoch_{ep}")

    for name, fn in [("on_train_start", on_train_start), ("on_train_epoch_start", on_train_epoch_start),
                     ("on_train_batch_start", on_train_batch_start), ("on_train_batch_end", on_train_batch_end),
                     ("on_fit_epoch_end", on_fit_epoch_end)]:
        model.add_callback(name, fn)

    status = "completed"
    try:
        if args.resume:
            opt12.event("train_resume", run_id=run_id, source=source.as_posix())
            model.train(resume=True)
        else:
            model.train(**ultra.train_kwargs(config, data_yaml, output, args.device))
    except StopRun as stop:
        status = stop.status
    seconds = time.perf_counter() - t0
    proc.update(status=status, seconds=round(seconds, 1), ended=opt12.stamp())
    if args.device != "cpu":
        proc["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(int(args.device))
        proc["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(int(args.device))
    obs["epoch_checkpoints"] = sorted(p.name for p in weights_dir.glob("epoch*.pt"))
    opt12.save_json(obs_path, obs)
    import shutil

    copies = results / "ultralytics"
    copies.mkdir(exist_ok=True)
    for name in ("results.csv", "args.yaml"):
        if (output / "train" / name).exists():
            shutil.copy2(output / "train" / name, copies / name)
    opt12.event("train_process_end", run_id=run_id, status=status, seconds=round(seconds, 1))
    print(json.dumps({"status": status, "minutes": round(seconds / 60, 1)}))
    if status.startswith("catastrophe"):
        sys.exit(3)


if __name__ == "__main__":
    main()
