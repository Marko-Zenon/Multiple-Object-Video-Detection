"""OPT-08 closeout robustness: leave-one-clip-out val_full mAP of x2 and every upload-ready candidate, and a
re-hash of every upload-ready CSV and its test predictions against its run record. Evaluation only (project
evaluator, stored predictions); nothing is selected or retuned from these numbers.

    python scripts/opt08_robustness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import opt08_loop as L

OUT = L.EVID / "robustness"


def candidates() -> dict[str, Path]:
    out = {"x2": L.X2["val_full"][0]}
    for run in sorted(L.EVID.glob("*/test/run.json")):
        exp = run.parent.parent.name
        cid = L.load_json(run)["candidate"]
        out[f"{exp}:{cid}"] = L.OUT / exp / cid / "val_full.json"
    return out


def main() -> None:
    subsets = L.subsets()
    clips = sorted(k[5:] for k in subsets if k.startswith("clip:"))
    full = subsets["val_full"]
    gt_clip = {i: c for c in clips for i in subsets[f"clip:{c}"]}
    jobs = []
    for name, path in candidates().items():
        for c in clips:
            ids = [i for i in full if gt_clip[i] != c]
            jobs.append((path, f"{name}|without:{c}", ids))
    results = L.eval_paths(jobs, workers=4)
    rows = []
    for name in candidates():
        if name == "x2":
            continue
        for c in clips:
            base = results[f"x2|without:{c}"]["metrics"]["mAP"]
            cand = results[f"{name}|without:{c}"]["metrics"]["mAP"]
            rows.append(
                {
                    "candidate": name,
                    "left_out_clip": c,
                    "x2_mAP": base,
                    "cand_mAP": cand,
                    "delta": cand - base,
                }
            )
    L.write_rows(OUT / "leave_one_clip_out.csv", rows)
    summary = {}
    for name in {r["candidate"] for r in rows}:
        ds = [r["delta"] for r in rows if r["candidate"] == name]
        summary[name] = {
            "min_delta": min(ds),
            "max_delta": max(ds),
            "all_positive": all(d > 0 for d in ds),
        }
    hashes = {}
    for run in sorted(L.EVID.glob("*/test/run.json")):
        r = L.load_json(run)
        csv, pred = Path(r["csv"]), Path(r["test_predictions"])
        hashes[run.parent.parent.name] = {
            "csv": r["csv"],
            "csv_sha256_recorded": r["csv_sha256"],
            "csv_sha256_now": L.sha256(csv),
            "test_predictions_sha256_now_matches": L.sha256(pred)
            == r["test_predictions_sha256"],
            "uploaded": r["uploaded"],
        }
        hashes[run.parent.parent.name]["csv_identical"] = (
            hashes[run.parent.parent.name]["csv_sha256_now"] == r["csv_sha256"]
        )
    L.save_json(
        OUT / "summary.json",
        {
            "leave_one_clip_out": summary,
            "artifact_rehash": hashes,
            "at_kyiv": L.o7.kyiv_now(),
        },
    )
    print(summary)
    print(
        {
            k: (v["csv_identical"], v["test_predictions_sha256_now_matches"])
            for k, v in hashes.items()
        }
    )


if __name__ == "__main__":
    main()
