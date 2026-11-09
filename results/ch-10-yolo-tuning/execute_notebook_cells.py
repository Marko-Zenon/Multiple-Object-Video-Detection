"""Execute only the CH-10 notebook cells in a fresh kernel and write their outputs back (run under the notebook lock).

    python results/ch-10-yolo-tuning/execute_notebook_cells.py

The three CH-10 cells are copied into a temporary notebook and run with nbclient in a clean kernel from the repo
root, so no legacy cell (which would retrain models) is executed. Their outputs are then stored in the CH-10 code
cell of homework.ipynb. Every other cell is checked to be unchanged, and the result is validated with nbformat.
Writes results/ch-10-yolo-tuning/notebook_validation.json.
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
IDS = ("ch10-summary", "ch10-evidence", "ch10-student-todos")
REPORT = ROOT / "results" / "ch-10-yolo-tuning" / "notebook_validation.json"


def digest(cell: dict) -> str:
    return hashlib.sha256(json.dumps(cell, sort_keys=True).encode()).hexdigest()


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_bytes().replace(b"\r\n", b"\n"))
    before = {i: digest(c) for i, c in enumerate(notebook["cells"]) if c.get("id") not in IDS}
    ours = [c for c in notebook["cells"] if c.get("id") in IDS]
    if [c["id"] for c in ours] != list(IDS):
        raise SystemExit("CH-10 cells missing; run notebook_cells.py first")
    temp = nbformat.from_dict(
        {**{k: v for k, v in notebook.items() if k != "cells"}, "cells": copy.deepcopy(ours)}
    )
    NotebookClient(
        temp, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    ).execute()
    code = next(c for c in temp.cells if c["id"] == "ch10-evidence")
    errors = [o for o in code["outputs"] if o.get("output_type") == "error"]
    if errors:
        raise SystemExit(f"CH-10 cell raised: {errors[0].get('ename')}: {errors[0].get('evalue')}")
    for cell in notebook["cells"]:
        if cell.get("id") == "ch10-evidence":
            cell["outputs"] = json.loads(json.dumps(code["outputs"]))
            cell["execution_count"] = code["execution_count"]
    after = {i: digest(c) for i, c in enumerate(notebook["cells"]) if c.get("id") not in IDS}
    if after != before:
        raise SystemExit("a non-CH-10 cell changed; nothing written")
    nbformat.validate(nbformat.from_dict(notebook))
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    images = sum("image/png" in o.get("data", {}) for o in code["outputs"])
    report = {
        "cells_total": len(notebook["cells"]),
        "cells_unchanged_outside_ch10": len(before),
        "ch10_cells": list(IDS),
        "executed": "only the CH-10 cells, fresh python3 kernel (nbclient), cwd = repo root",
        "errors": 0,
        "outputs": len(code["outputs"]),
        "embedded_png": images,
        "nbformat_valid": True,
        "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
