"""Append the CH-14 section to homework.ipynb, then run only its cells (under the notebook lock).

    python results/ch-14-final/notebook_cells.py            # append 3 cells if absent, execute them, validate

Every existing cell, output and the notebook metadata stay unchanged as JSON objects (checked by hash). The CH-14
code cell is self-contained (own imports) and reads only tracked files in results/ and configs/. Only the CH-14
cells run, in a fresh python3 kernel (nbclient) from the repo root, so no legacy training cell is executed.
Writes results/ch-14-final/notebook_validation.json.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "homework.ipynb"
IDS = ("ch14-summary", "ch14-evidence", "ch14-student-todos")
REPORT = ROOT / "results" / "ch-14-final" / "notebook_validation.json"

SUMMARY = """## CH-14 — Final training, final comparison and the test submission

Sources: `results/final_rtdetr/` (the training run), `results/ch-14-final/` (evaluation, comparison, decision, speed, submission
checks) and `results/submissions.csv`. The cell below only reads stored evidence. Every AP comes from the project evaluator
(val_full, 2,846 frames, maxDets 500).

- **Training (`configs/runs/final_rtdetr.json` = `configs/runs/final.json` + `train_split`).** RT-DETR-l from the official
  COCO-pretrained `rtdetr-l.pt`, the `rtdetr_baseline` settings, trained on `train_final` (every 4th frame of each of the 56 train
  clips, 6,072 frames), 6 epochs = 36,432 images, imgsz 1344, batch 1, AMP, seed 42, flip only, the CH-02 ignore masking.
- **Two inference modes on val_full.** The unchanged single pass (as in CH-11) and the frozen CH-13 pipeline
  (`configs/runs/final_inference.json`: full + flip + 2x2 tiles, class-wise NMS 0.5, top 500).
- **Decision rule.** `results/ch-14-final/selection_rule.md` was committed before any CH-14 result. The candidate is `last.pt`
  (epoch 6). It replaces the fallback (CH-11 `rtdetr_baseline` + CH-13 pipeline, val_full mAP 0.2696) only if its lead is at least
  the run-to-run spread 0.0063, it wins at least 5 of 9 secondary checks, and it keeps AP_small."""

CODE = r"""import json
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

root = Path.cwd()
ch14 = root / "results" / "ch-14-final"

display(Markdown("**Per-epoch val_fast curve (single pass; reported, does not choose the checkpoint)**"))
display(pd.read_csv(ch14 / "epochs.csv"))
display(Image(filename=str(ch14 / "figures" / "curve.png")))

display(Markdown("**val_full: CH-14 vs the fallback, both inference modes**"))
display(pd.read_csv(ch14 / "comparison.csv").set_index("model"))
per_class = pd.read_csv(ch14 / "per_class.csv")
display(per_class.pivot(index="name", columns="model", values="AP").round(4))
display(Image(filename=str(ch14 / "figures" / "per_class.png")))
display(Image(filename=str(ch14 / "figures" / "density.png")))

selection = json.loads((ch14 / "selection.json").read_text())
display(Markdown(f"**Rule B:** mAP C {selection['mAP']['C']:.4f} vs F {selection['mAP']['F']:.4f} "
                 f"(lead {selection['mAP']['lead']:+.4f}, spread {selection['spread']}); secondary checks won "
                 f"{selection['secondary_wins']}/9; criteria {selection['criteria']}; "
                 f"**primary submission: {selection['primary_submission']}**"))
display(pd.DataFrame(selection["secondary_checks"]).round(4))

speed = json.loads((ch14 / "speed" / "session.json").read_text())
display(Markdown("**Same-session RTX 5070 speed (first 30 val_full frames preloaded, FP32, batch 1)**"))
display(pd.DataFrame(speed["rows"])[["name", "mode", "fps", "passes_per_frame", "test_minutes_3811_frames",
                                     "peak_cuda_allocated_bytes"]].round(2))
runtime = json.loads((root / "results" / "final_rtdetr" / "runtime.json").read_text())
display(Markdown(f"**Training wall-clock:** {runtime['train_seconds_total'] / 3600:.2f} h over "
                 f"{len(runtime['train_process_seconds'])} process(es)"))

display(Markdown("**Kaggle uploads (`results/submissions.csv`)**"))
display(pd.read_csv(root / "results" / "submissions.csv")[["submission_file", "val_full_mAP", "kaggle_public_score",
                                                           "kaggle_status"]])
"""

TODOS = """### CH-14 — student discussion TODOs

- Interpret the data-strategy result: what did 2.17x more distinct training frames change (single pass and CH-13 pipeline), and
  in which classes, sizes and density bins? Use `comparison.csv`, `per_class.csv` and `density.png`.
- Discuss the per-epoch val_fast curve compared with `rtdetr_baseline` (does the larger list change where the curve peaks?).
- Explain the final Kaggle selection (two selections, chosen by internal validation) in your own words, and compare the public
  score with the val_full ranking. Conclusions are written by the students."""


def cell(cell_type: str, cell_id: str, source: str) -> dict:
    out = {
        "cell_type": cell_type,
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        out.update(execution_count=None, outputs=[])
    return out


def digest(c: dict) -> str:
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_bytes().replace(b"\r\n", b"\n"))
    existing = [c for c in notebook["cells"] if c.get("id") not in IDS]
    before = [digest(c) for c in existing]
    if not any(c.get("id") in IDS for c in notebook["cells"]):
        notebook["cells"] += [
            cell("markdown", IDS[0], SUMMARY),
            cell("code", IDS[1], CODE),
            cell("markdown", IDS[2], TODOS),
        ]
    ours = [c for c in notebook["cells"] if c.get("id") in IDS]
    temp = nbformat.reads(
        json.dumps(
            {
                **{k: v for k, v in notebook.items() if k != "cells"},
                "cells": copy.deepcopy(ours),
            }
        ),
        as_version=4,
    )
    NotebookClient(
        temp,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    code = next(c for c in temp.cells if c["id"] == IDS[1])
    errors = [o for o in code["outputs"] if o.get("output_type") == "error"]
    if errors:
        raise SystemExit(
            f"CH-14 cell raised: {errors[0].get('ename')}: {errors[0].get('evalue')}"
        )
    for c in notebook["cells"]:
        if c.get("id") == IDS[1]:
            c["outputs"] = json.loads(json.dumps(code["outputs"]))
            c["execution_count"] = code["execution_count"]
    after = [digest(c) for c in notebook["cells"] if c.get("id") not in IDS]
    if after != before:
        raise SystemExit("a non-CH-14 cell changed; nothing written")
    nbformat.validate(nbformat.from_dict(notebook))
    NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "cells_total": len(notebook["cells"]),
        "cells_unchanged_outside_ch14": len(before),
        "ch14_cells": list(IDS),
        "executed": "only the CH-14 cells, fresh python3 kernel (nbclient), cwd = repo root",
        "errors": 0,
        "outputs": len(code["outputs"]),
        "embedded_png": sum("image/png" in o.get("data", {}) for o in code["outputs"]),
        "nbformat_valid": True,
        "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
