"""CH-06 check: execute Stage 1 of homework.ipynb (every cell before "## 5. Faster R-CNN") in a clean kernel.

Sections 5-6 are the legacy v0 training runs and are never executed (training never runs in the notebook).
Usage (repo root):  python results/ch-06-eda/run_stage1.py          # check only, the notebook is not changed
                    python results/ch-06-eda/run_stage1.py --write  # also store the fresh Stage 1 outputs
"""
import copy
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "homework.ipynb"

nb = nbformat.read(NOTEBOOK, as_version=4)
end = next(i for i, c in enumerate(nb.cells)
           if c.cell_type == "markdown" and c.source.lstrip().startswith("## 5. Faster R-CNN"))
stage = copy.deepcopy(nb)
stage.cells = stage.cells[:end]
for cell in stage.cells:
    if cell.cell_type == "code":
        cell.outputs, cell.execution_count = [], None

start = time.time()
NotebookClient(stage, timeout=1800, kernel_name="python3", record_timing=False,
               resources={"metadata": {"path": str(REPO)}}).execute()  # raises on the first cell error
code = [c for c in stage.cells if c.cell_type == "code"]
print(f"Stage 1: {end} cells ({len(code)} code) executed without errors in {time.time() - start:.0f} s")

if "--write" in sys.argv:
    nb.cells[:end] = stage.cells
    nbformat.write(nb, NOTEBOOK)
    print("fresh Stage 1 outputs written to", NOTEBOOK.name)
