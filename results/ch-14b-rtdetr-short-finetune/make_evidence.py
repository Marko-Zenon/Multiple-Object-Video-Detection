"""Build the CH-14b tables and the rule-R decisions from stored results (CPU only, nothing recomputed on the GPU).

    python results/ch-14b-rtdetr-short-finetune/make_evidence.py

Reads results/ch-14b-rtdetr-short-finetune/<candidate>/<split>/{single,pipeline}/metrics.json for every candidate
that was evaluated, plus the fallback F evidence of CH-13 (val_fast and val_full, CH-13 pipeline) and CH-11 (single
pass). Applies selection_rule.md as written (Stage A gate A1, rule R, Stage B epoch choice) and writes:
stages.csv (every evaluation), comparison_/per_class_/density_<candidate>.csv (val_full, candidate vs F) and
selection.json (the decisions).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = REPO / "results"
SPREAD = 0.0063  # CH-10 replicate spread (CH-12 rule 3), unchanged from CH-14 rule B
APS_GUARD = 0.003  # CH-13 small-object guard, unchanged
A1_GATE = 0.2725  # F on val_fast, CH-13 pipeline (configs/runs/final_inference.json evidence.val_fast.selected)
SECONDARY = ["AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
METRICS = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
STAGE_A = "stageA_epoch1"
STAGE_B = ["stageB_epoch1", "stageB_epoch2"]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def evaluation(name: str, split: str, mode: str = "pipeline") -> dict | None:
    path = HERE / name / split / mode / "metrics.json"
    return load(path) if path.exists() else None


def fallback(split: str, mode: str = "pipeline") -> dict:
    if mode == "single":
        if split == "val_full":
            return load(RES / "rtdetr_baseline" / "metrics.json")
        return {
            "metrics": {
                "mAP": load(RES / "rtdetr_baseline" / "curve.json")[-1]["val_fast"][
                    "mAP"
                ]
            }
        }
    return load(
        RES / f"ch-13-inference/runs/{split}/combo_flip_tile_nms05/metrics.json"
    )


def rule_r(c: dict, f: dict, identity_ok: bool) -> dict:
    checks = [{"check": k, "X": c["metrics"][k], "F": f["metrics"][k], "X_leads": c["metrics"][k] > f["metrics"][k]}
              for k in SECONDARY]  # fmt: skip
    for cb, fb in zip(c["density"], f["density"], strict=True):
        assert cb["bin"] == fb["bin"]
        checks.append(
            {
                "check": f"density {cb['bin']}",
                "X": cb["mAP"],
                "F": fb["mAP"],
                "X_leads": cb["mAP"] > fb["mAP"],
            }
        )
    lead = c["metrics"]["mAP"] - f["metrics"]["mAP"]
    wins = sum(x["X_leads"] for x in checks)
    criteria = {
        "R1_mAP_lead_ge_spread": lead >= SPREAD,
        "R2_secondary_wins_ge_5_of_9": wins >= 5,
        "R3_AP_small_guard": c["metrics"]["AP_small"]
        >= f["metrics"]["AP_small"] - APS_GUARD,
        "R4_identity": identity_ok,
    }
    return {
        "mAP": {"X": c["metrics"]["mAP"], "F": f["metrics"]["mAP"], "lead": lead},
        "spread": SPREAD,
        "secondary_checks": checks,
        "secondary_wins": wins,
        "checks_lost": [x["check"] for x in checks if not x["X_leads"]],
        "criteria": criteria,
        "tie": 0 < lead < SPREAD,
        "passes": all(criteria.values()),
    }


def identity_ok(stage: str) -> bool:
    path = HERE / f"{stage}_identity.json"
    return path.exists() and load(path)["passed"]


def comparison_tables(name: str) -> None:
    c, f = evaluation(name, "val_full"), fallback("val_full")
    rows = [{"model": "F rtdetr_baseline + CH-13 pipeline", **{k: round(f["metrics"][k], 4) for k in METRICS}},
            {"model": f"{name} + CH-13 pipeline", **{k: round(c["metrics"][k], 4) for k in METRICS}},
            {"model": "delta", **{k: round(c["metrics"][k] - f["metrics"][k], 4) for k in METRICS}}]  # fmt: skip
    write_csv(HERE / f"comparison_{name}.csv", rows)
    per_class = []
    for xc, fc in zip(c["per_class"], f["per_class"], strict=True):
        assert xc["class_id"] == fc["class_id"]
        per_class.append({"class_id": xc["class_id"], "name": xc["name"], "num_gt": xc["num_gt"],
                          "AP_F": round(fc["AP"], 4), f"AP_{name}": round(xc["AP"], 4),
                          "delta": round(xc["AP"] - fc["AP"], 4)})  # fmt: skip
    write_csv(HERE / f"per_class_{name}.csv", per_class)
    density = []
    for xb, fb in zip(c["density"], f["density"], strict=True):
        density.append({"bin": xb["bin"], "num_images": xb["num_images"], "num_gt": xb["num_gt"],
                        "mAP_F": round(fb["mAP"], 4), f"mAP_{name}": round(xb["mAP"], 4),
                        "delta": round(xb["mAP"] - fb["mAP"], 4)})  # fmt: skip
    write_csv(HERE / f"density_{name}.csv", density)


def main() -> None:
    stages = []
    for name in [STAGE_A, *STAGE_B]:
        for split in ("val_fast", "val_full"):
            for mode in ("single", "pipeline"):
                m = evaluation(name, split, mode)
                if m:
                    stages.append({"candidate": name, "split": split, "mode": mode,
                                   **{k: round(m["metrics"][k], 4) for k in METRICS}})  # fmt: skip
    for split in ("val_fast", "val_full"):
        f = fallback(split)
        stages.append({"candidate": "F rtdetr_baseline", "split": split, "mode": "pipeline",
                       **{k: round(f["metrics"][k], 4) for k in METRICS}})  # fmt: skip
    write_csv(HERE / "stages.csv", stages)

    decision = {
        "rule": "results/ch-14b-rtdetr-short-finetune/selection_rule.md",
        "selected": "F",
    }
    a_fast = evaluation(STAGE_A, "val_fast")
    a = {"identity_passed": identity_ok("stageA"), "val_fast_pipeline_mAP": a_fast and a_fast["metrics"]["mAP"],
         "A1_gate": A1_GATE}  # fmt: skip
    a["A1_competitive"] = bool(a_fast) and a_fast["metrics"]["mAP"] >= A1_GATE
    if a["A1_competitive"] and evaluation(STAGE_A, "val_full"):
        a["rule_R"] = rule_r(
            evaluation(STAGE_A, "val_full"), fallback("val_full"), a["identity_passed"]
        )
        if a["rule_R"]["passes"]:
            decision["selected"] = STAGE_A
        comparison_tables(STAGE_A)
    decision["stage_A"] = a

    b_fast = {n: evaluation(n, "val_fast") for n in STAGE_B}
    if all(b_fast.values()):
        e1, e2 = (b_fast[n]["metrics"]["mAP"] for n in STAGE_B)
        chosen = STAGE_B[0] if e1 > e2 else STAGE_B[1]  # an exact tie picks epoch 2
        b = {"val_fast_pipeline_mAP": dict(zip(STAGE_B, (e1, e2), strict=True)), "chosen_by_val_fast": chosen,
             "identity_passed": identity_ok("stageB")}  # fmt: skip
        if evaluation(chosen, "val_full"):
            b["rule_R"] = rule_r(
                evaluation(chosen, "val_full"),
                fallback("val_full"),
                b["identity_passed"],
            )
            if b["rule_R"]["passes"] and decision["selected"] == "F":
                decision["selected"] = chosen
            comparison_tables(chosen)
        decision["stage_B"] = b
    decision["primary_submission"] = (
        "fallback rtdetr_baseline + CH-13 pipeline (unchanged CSV 73f73e21...793b)"
        if decision["selected"] == "F"
        else f"{decision['selected']} + CH-13 pipeline"
    )
    (HERE / "selection.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in decision.items() if k != "stage_A" and k != "stage_B"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
