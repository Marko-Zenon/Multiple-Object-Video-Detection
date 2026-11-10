"""Append the CH-14b section to homework.ipynb, then run only its cells (under the notebook lock).

    python results/ch-14b-rtdetr-short-finetune/notebook_cells.py   # append 3 cells if absent, execute them, validate

Same method as results/ch-14-final/notebook_cells.py: every existing cell, output and the notebook metadata stay
unchanged (checked by hash); the CH-14b code cell is self-contained and reads only tracked files in results/; only the
CH-14b cells run, in a fresh python3 kernel (nbclient) from the repo root. Writes notebook_validation.json next to it.
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
IDS = ("ch14b-summary", "ch14b-evidence", "ch14b-student-todos")
REPORT = ROOT / "results" / "ch-14b-rtdetr-short-finetune" / "notebook_validation.json"

SUMMARY = """## CH-14b — Authorized extension: the CH-14 epoch-1 checkpoint

After the competition deadline was extended, student Badabuba authorized one limited post-CH-14 experiment. The CH-14 section
above and `results/ch-14-final/` are unchanged. Sources: `results/ch-14b-rtdetr-short-finetune/` (rule, identity checks,
evaluations, decision, speed, submission checks). The cell below only reads stored evidence; every AP comes from the project
evaluator (maxDets 500) with the frozen CH-13 pipeline (`configs/runs/final_inference.json`).

- **Candidate (Stage A).** `epoch0.pt` of the CH-14 `final_rtdetr` run: RT-DETR-l after epoch 1 of 6 on `train_final` (6,072
  frames), sha256 `6ff4a430…5b0b`. The CH-14 single-pass val_fast curve peaked there (0.2790) and fell to 0.2381 at epoch 6.
- **Rule** (`selection_rule.md`, committed before any CH-14b result): val_fast gate (≥ 0.2725, the fallback on val_fast), then
  one val_full evaluation and the CH-14 rule B thresholds unchanged (lead ≥ 0.0063, ≥ 5 of 9 secondary checks, AP_small guard).
  The only deviation is the identity check: "6 completed epochs" is replaced by the exact checkpoint hash and train_final.
  val_fast is a subset of val_full, so a checkpoint picked on val_fast is slightly optimistic on val_full.
- **Stage B** (a 2-epoch fine-tune of the fallback) was to run only if Stage A lost; it did not run."""

CODE = r"""import json
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

ch14b = Path.cwd() / "results" / "ch-14b-rtdetr-short-finetune"
name = "stageA_epoch1"

display(Markdown("**Every CH-14b evaluation and the fallback F (CH-13 pipeline unless marked single)**"))
display(pd.read_csv(ch14b / "stages.csv"))

display(Markdown("**val_full: epoch-1 checkpoint vs the fallback (CH-13 pipeline)**"))
display(pd.read_csv(ch14b / f"comparison_{name}.csv").set_index("model"))
display(pd.read_csv(ch14b / f"per_class_{name}.csv").set_index("name"))
display(pd.read_csv(ch14b / f"density_{name}.csv").set_index("bin"))

selection = json.loads((ch14b / "selection.json").read_text())
rule = selection["stage_A"]["rule_R"]
display(Markdown(f"**Rule R:** mAP {rule['mAP']['X']:.4f} vs F {rule['mAP']['F']:.4f} (lead {rule['mAP']['lead']:+.4f}, "
                 f"spread {rule['spread']}); secondary checks won {rule['secondary_wins']}/9, lost {rule['checks_lost']}; "
                 f"criteria {rule['criteria']}; **internal selection: {selection['primary_submission']}**"))

speed = json.loads((ch14b / "speed" / "session.json").read_text())
display(Markdown("**Same-session RTX 5070 speed (first 30 val_full frames preloaded, FP32, batch 1)**"))
display(pd.DataFrame(speed["rows"])[["name", "mode", "fps", "passes_per_frame", "test_minutes_3811_frames",
                                     "peak_cuda_allocated_bytes"]].round(2))

sub = json.loads((ch14b / "final_submission.json").read_text())
display(Markdown(f"**Test candidate CSV (not uploaded by Claude, not in Git):** `{sub['csv_file']}`, {sub['csv_bytes']:,} bytes, "
                 f"sha256 `{sub['csv_sha256']}`, {sub['rows']} rows, {sub['detections']:,} detections; validators: "
                 f"{sub['validators']}"))
"""

TODOS = """### CH-14b — student TODOs

- Decide whether to upload the CH-14b candidate CSV and record its public score in `results/submissions.csv` (Claude does not upload).
- Make and record the two Kaggle final selections (G2c: by internal validation). The internal rule now ranks the CH-14b candidate
  first (val_full 0.2860 vs 0.2696); the public score is recorded but does not override the rule.
- Interpret why epoch 1 of the train_final run beats epoch 6 and the train_rubric baseline (curve, classes, density bins, the one
  lost check `density >=100`), and how the val_fast-to-val_full selection bias affects the claim. Conclusions are written by the
  students."""


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
            f"CH-14b cell raised: {errors[0].get('ename')}: {errors[0].get('evalue')}"
        )
    for c in notebook["cells"]:
        if c.get("id") == IDS[1]:
            c["outputs"] = json.loads(json.dumps(code["outputs"]))
            c["execution_count"] = code["execution_count"]
    after = [digest(c) for c in notebook["cells"] if c.get("id") not in IDS]
    if after != before:
        raise SystemExit("a non-CH-14b cell changed; nothing written")
    nbformat.validate(nbformat.from_dict(notebook))
    NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "cells_total": len(notebook["cells"]),
        "cells_unchanged_outside_ch14b": len(before),
        "ch14b_cells": list(IDS),
        "executed": "only the CH-14b cells, fresh python3 kernel (nbclient), cwd = repo root",
        "errors": 0,
        "outputs": len(code["outputs"]),
        "nbformat_valid": True,
        "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
