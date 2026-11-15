"""OPT-12 weight averaging (AVG3, WiSE-FT W(alpha), soup w070), same arithmetic as OPT-08/09/11 soups.

Floating tensors are averaged in float32 and stored in their original dtype; non-float tensors come from the first
input. The inputs are the `ema` modules of Ultralytics checkpoints (what inference loads). The saved file is a copy of
the first checkpoint's metadata with the averaged `ema`, and `model/optimizer/scaler = None`, as in opt08_soup.py.
"""

from __future__ import annotations

import copy
from pathlib import Path


def weighted_state(states: list[dict], weights: list[float]) -> dict:
    out = {}
    for k, t0 in states[0].items():
        if t0.is_floating_point():
            acc = weights[0] * t0.float()
            for s, w in zip(states[1:], weights[1:]):
                acc = acc + w * s[k].float()
            out[k] = acc.to(t0.dtype)
        else:
            out[k] = t0.clone()
    return out


def build_soup(paths: list[Path], weights: list[float], out: Path, provenance: dict) -> dict:
    import torch

    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got {weights}")
    ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    states = [c["ema"].state_dict() for c in ckpts]
    names = list(states[0])
    for p, s in zip(paths, states):
        if list(s) != names or any(s[k].shape != states[0][k].shape for k in names):
            raise ValueError(f"{p}: tensor names / shapes differ from {paths[0]}")
    st = weighted_state(states, weights)
    model = copy.deepcopy(ckpts[0]["ema"])
    model.load_state_dict(st)
    first = ckpts[0]
    ckpt = {k: v for k, v in first.items() if k not in ("optimizer", "scaler", "ema", "model")}
    ckpt.update({"ema": model, "model": None, "optimizer": None, "scaler": None, "opt12_soup": provenance})
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    return {"num_tensors": len(st), "dtypes_equal": all(model.state_dict()[k].dtype == states[0][k].dtype for k in names)}
