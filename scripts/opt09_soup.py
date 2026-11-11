"""OPT-09 Phase B: nearby untested soup weights w x Model A (epoch0.pt) + (1 - w) x epoch1.pt. Inference only.

    python scripts/opt09_soup.py build                      # compatibility + one checkpoint per frozen weight
    python scripts/opt09_soup.py smoke                      # w080 identity (weights + 20-frame GPU raw) + 20-frame smoke
    python scripts/opt09_soup.py cache W --stage discovery  # raw 6-view RT-DETR output per clip (GPU, resumable)

The soup arithmetic, the compatibility check, the view list and the raw format are OPT-08's (`scripts/opt08_soup.py`,
reused unchanged). The protected E6 soup w080 and its raw caches are read from the OPT-08 worktree, never written.
Frozen weights: `movd.opt09.SOUP_WEIGHTS` / GRIDS["b_soup_weight"]; rule: results/opt-09-e6-tuning/selection_rule.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import opt07_loop as o7
import opt08_soup as s8

from movd import infer, opt09, rtdetr

HW2 = REPO.parent
OPT08_SOUP = HW2 / "MOVD-opt08" / "outputs" / "opt08" / "soup"
E6_SOUP = (
    OPT08_SOUP / "soup_w080.pt",
    "2d60d3fa66d9fd37032bcaaaeb297e98510217b600b5809f73873b36360cd939",
)
OUT = REPO / "outputs" / "opt09" / "soup"
EVID = REPO / "results" / "opt-09-e6-tuning" / "b_soup_weight"
NEW = [t for t in opt09.SOUP_WEIGHTS if t != opt09.E6["soup"]]


def soup_path(tag: str) -> Path:
    return E6_SOUP[0] if tag == opt09.E6["soup"] else OUT / f"soup_{tag}.pt"


def weights_path(tag: str) -> Path:
    """The checkpoint of a soup tag, hash-checked (w080 against OPT-08, new soups against OPT-09's build record)."""
    if tag == opt09.E6["soup"]:
        expected = E6_SOUP[1]
    else:
        expected = o7.load_json(EVID / "soup_build.json")[tag]["sha256"]
    path = soup_path(tag)
    if o7.sha256(path) != expected:
        raise SystemExit(f"{path}: sha256 differs from the build record")
    return path


def raw_dir(tag: str, stage: str) -> Path:
    if tag == opt09.E6["soup"]:
        return OPT08_SOUP / "raw" / tag / stage
    return OUT / "raw" / tag / stage


def merged(tag: str, stages: list[str], factor: float, margin: float) -> list[dict]:
    """CH-13 merge (with the border rule) of every cached clip of `stages`; each stage must be complete."""
    wh = s8.sizes("test" if stages == ["test"] else "val")
    preds = []
    for stage in stages:
        for clip in sorted(s8.stage_frames(stage)):
            shard = raw_dir(tag, stage) / f"{clip}.npz"
            if not shard.exists():
                raise SystemExit(f"{shard} missing: cache {tag} {stage} first")
            preds.extend(s8.merge_shard(shard, wh, factor, margin))
    return preds


def cmd_build(_args) -> None:
    import torch

    compat = s8.compatibility()
    if not compat["compatible"]:
        raise SystemExit("compatibility imperfect: Phase B rejected (rule)")
    a, b = s8.load_ckpt(*s8.A), s8.load_ckpt(*s8.B)
    sa, sb = a["ema"].state_dict(), b["ema"].state_dict()
    # identity: OPT-08's soup arithmetic at w = 0.8 must give E6's checkpoint tensors exactly
    e6 = torch.load(weights_path("w080"), map_location="cpu", weights_only=False)
    rebuilt = s8.soup_state(sa, sb, opt09.SOUP_WEIGHTS["w080"])
    se6 = e6["ema"].state_dict()
    identity = list(rebuilt) == list(se6) and all(
        torch.equal(rebuilt[k], se6[k]) for k in se6
    )
    if not identity:
        raise SystemExit("rebuilt w080 differs from the E6 soup: stop")
    record = {
        "compatibility": compat,
        "w080_rebuilt_tensor_identical_to_E6": identity,
        "E6_soup": {"path": o7.rel(E6_SOUP[0]), "sha256": E6_SOUP[1]},
    }
    for tag in NEW:
        w = opt09.SOUP_WEIGHTS[tag]
        model = copy.deepcopy(a["ema"])
        state = s8.soup_state(sa, sb, w)
        model.load_state_dict(state)
        names_shapes_ok = list(model.state_dict()) == list(sa) and all(
            model.state_dict()[k].shape == sa[k].shape
            and model.state_dict()[k].dtype == sa[k].dtype
            for k in sa
        )
        ckpt = {
            k: v
            for k, v in a.items()
            if k not in ("optimizer", "scaler", "ema", "model")
        }
        ckpt.update({"ema": model, "model": None, "optimizer": None, "scaler": None})
        ckpt["opt09_soup"] = {
            "weight_model_A": w,
            "model_A_sha256": s8.A[1],
            "adjacent_sha256": s8.B[1],
        }
        path = soup_path(tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)
        record[tag] = {
            "weight_model_A": w,
            "path": o7.rel(path),
            "sha256": o7.sha256(path),
            "names_shapes_dtypes_equal_model_A": names_shapes_ok,
            "num_tensors": len(state),
        }
        if not names_shapes_ok:
            raise SystemExit(f"{tag}: parameter names / shapes differ: stop")
    o7.save_json(EVID / "soup_build.json", record)
    print(
        json.dumps({k: v for k, v in record.items() if k != "compatibility"}, indent=2)
    )


def cache(
    tag: str,
    stage: str,
    device: str = "0",
    max_frames: int = 0,
    clips: list[str] | None = None,
    out_dir: Path | None = None,
) -> dict:
    """OPT-08's raw cache (`movd.infer.raw_pass` per CH-13 view) for a new soup; never writes OPT-08 shards."""
    import cv2
    import torch

    if tag == opt09.E6["soup"] and out_dir is None:
        raise SystemExit(
            "the E6 raw caches are protected (OPT-08): use them, do not rebuild"
        )
    cfg = s8.config()
    specs = infer.pass_specs(cfg)
    grid, frac = cfg.get("tile_grid", 2), cfg.get("tile_frac", 0.6)
    by_clip = s8.stage_frames(stage)
    model = rtdetr.load_model(weights_path(tag), "rtdetr")
    torch.zeros(1, device=f"cuda:{device}")
    out_dir = out_dir or raw_dir(tag, stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    seconds, computed = 0.0, 0
    for clip in clips or sorted(by_clip):
        shard = out_dir / f"{clip}.npz"
        if shard.exists():
            continue
        ids, boxes, scores = [], [], []
        start = time.perf_counter()
        for image_id, path in by_clip[clip][: max_frames or None]:
            image = cv2.imread(str(path))
            if image is None:
                raise OSError(f"cannot read {path}")
            raws = [
                infer.raw_pass(model, image, s["imgsz"], device, s["view"], grid, frac)
                for s in specs
            ]
            ids.append(image_id)
            boxes.append(np.stack([r.boxes for r in raws]))
            scores.append(np.stack([r.scores for r in raws]))
        seconds += time.perf_counter() - start
        computed += len(ids)
        tmp = shard.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            image_ids=np.array(ids, np.int64),
            boxes=np.stack(boxes),
            scores=np.stack(scores),
        )
        tmp.replace(shard)
        print(
            f"{tag} {stage} {clip}: {len(ids)} frames, {seconds / 60:.1f} min",
            flush=True,
        )
    return {
        "frames": computed,
        "seconds": round(seconds, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }


def cmd_smoke(_args) -> None:
    """20 frames of the first discovery clip: w080 must equal its OPT-08 raw shard exactly; new soups finite."""
    clip = min(s8.stage_frames("discovery"))
    out = {"clip": clip}
    for tag in [opt09.E6["soup"], *NEW]:
        d = OUT / "raw_smoke" / tag
        shard = d / f"{clip}.npz"
        if shard.exists():
            shard.unlink()  # only this command's own earlier smoke output
        info = cache(tag, "discovery", max_frames=20, clips=[clip], out_dir=d)
        z = np.load(shard)
        rec = {
            **info,
            "finite": bool(
                np.isfinite(z["boxes"]).all() and np.isfinite(z["scores"]).all()
            ),
            "scores_in_0_1": bool(
                (z["scores"] >= 0).all() and (z["scores"] <= 1).all()
            ),
            "shape": list(z["boxes"].shape),
        }
        if tag == opt09.E6["soup"]:
            ref = np.load(raw_dir(tag, "discovery") / f"{clip}.npz")
            n = len(z["image_ids"])
            rec["identical_to_opt08_shard"] = bool(
                np.array_equal(ref["image_ids"][:n], z["image_ids"])
                and np.array_equal(ref["boxes"][:n], z["boxes"])
                and np.array_equal(ref["scores"][:n], z["scores"])
            )
        out[tag] = rec
    out["ok"] = out[opt09.E6["soup"]]["identical_to_opt08_shard"] and all(
        out[t]["finite"] and out[t]["scores_in_0_1"] for t in NEW
    )
    o7.save_json(EVID / "smoke.json", out)
    print(json.dumps(out, indent=2))
    if not out["ok"]:
        raise SystemExit("smoke failed: no full inference")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["build", "smoke", "cache"])
    ap.add_argument("tag", nargs="?", choices=NEW)
    ap.add_argument("--stage", choices=["discovery", "confirmation", "test"])
    args = ap.parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "smoke":
        cmd_smoke(args)
    else:
        if not (args.tag and args.stage):
            ap.error("cache needs a tag and --stage")
        print(json.dumps(cache(args.tag, args.stage)))


if __name__ == "__main__":
    main()
