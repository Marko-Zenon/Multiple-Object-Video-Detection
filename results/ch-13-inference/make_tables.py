"""Build the CH-13 tables, the rule-based selection and the figures from the stored run files (CPU, deterministic).

Inputs (all tracked): variants.json, runs/<split>/<variant>/{metrics,pipeline}.json, speed/<variant>.json,
passes/*.json. Outputs: comparison.csv, per_class.csv, density.csv, selection.json and figures/*.png.
`--tables-only` skips the figures. Run from anywhere: `python results/ch-13-inference/make_tables.py`.

The selection applies selection_rule.md (committed before any result) mechanically:
rule 2 (margin), rule 3 (speed), rule 4 (axis winners), rule 5 (greedy stack over the evaluated combinations),
rule 6 (ensemble margin) and rule 7 (val_full veto).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARGIN = 0.003  # rule 2
AP_SMALL_TOLERANCE = 0.003  # rule 2
ENSEMBLE_MARGIN = 0.005  # rule 6
STACK_MARGIN = 0.002  # rule 5
MIN_FPS = 1.6  # rule 3
REF = "ref"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def variants() -> tuple[list[dict], list[dict]]:
    doc = load(HERE / "variants.json")
    return doc["variants"], doc["combinations"]


def run(split: str, name: str) -> dict | None:
    d = HERE / "runs" / split / name
    if not (d / "metrics.json").exists():
        return None
    return {"metrics": load(d / "metrics.json"), "pipeline": load(d / "pipeline.json")}


def speed(name: str) -> dict | None:
    p = HERE / "speed" / f"{name}.json"
    return load(p) if p.exists() else None


def r4(x):
    return None if x is None else round(x, 4)


def comparison_rows(singles, combos) -> list[dict]:
    rows = []
    for split in ("val_fast", "val_full"):
        ref = run(split, REF)
        for v in singles + combos:
            r = run(split, v["name"])
            if r is None:
                continue
            m, p = r["metrics"]["metrics"], r["pipeline"]
            s = speed(v["name"])
            row = {
                "split": split,
                "variant": v["name"],
                "axis": v["axis"],
                "mAP": r4(m["mAP"]),
                "AP50": r4(m["AP50"]),
                "AP75": r4(m["AP75"]),
                "AP_small": r4(m["AP_small"]),
                "AP_medium": r4(m["AP_medium"]),
                "AP_large": r4(m["AP_large"]),
                "mAP_maxdets100": r4(r["metrics"]["metrics_maxdets100"]["mAP"]),
                "delta_mAP_vs_ref": r4(m["mAP"] - ref["metrics"]["metrics"]["mAP"])
                if ref
                else None,
                "delta_AP_small_vs_ref": r4(
                    m["AP_small"] - ref["metrics"]["metrics"]["AP_small"]
                )
                if ref
                else None,
                "passes_per_frame": p["num_raw_passes_per_frame"],
                "detections_per_frame_mean": p["detections_per_frame_mean"],
                "detections_per_frame_max": p["detections_per_frame_max"],
                "frames_at_500_cap": p["frames_at_500_cap"],
                "num_detections": p["num_detections"],
                "cpu_postprocess_ms_per_frame": p["cpu_postprocess_ms_per_frame"],
                "fps_end_to_end": r4(s["fps"]) if s else None,
                "test_minutes_3811_frames": s["test_minutes_3811_frames"]
                if s
                else None,
                "metrics_file": f"runs/{split}/{v['name']}/metrics.json",
            }
            for d in r["metrics"]["density"]:
                row[f"mAP_density_{d['bin']}"] = r4(d["mAP"])
            rows.append(row)
    return rows


def passes(row: dict, ensemble: bool = False) -> bool:
    margin = ENSEMBLE_MARGIN if ensemble else MARGIN
    ok = (
        row["delta_mAP_vs_ref"] >= margin
        and row["delta_AP_small_vs_ref"] >= -AP_SMALL_TOLERANCE
    )
    if row["fps_end_to_end"] is not None:
        ok = ok and row["fps_end_to_end"] >= MIN_FPS
    return ok


def select(rows, singles, combos) -> dict:
    fast = {r["variant"]: r for r in rows if r["split"] == "val_fast"}
    full = {r["variant"]: r for r in rows if r["split"] == "val_full"}
    screened, winners = [], {}
    for v in singles:
        if v["name"] == REF or v["name"] not in fast:
            continue
        row = fast[v["name"]]
        ok = passes(row, ensemble=v["axis"].startswith("5"))
        screened.append(
            {
                "variant": v["name"],
                "axis": v["axis"],
                "delta_mAP": row["delta_mAP_vs_ref"],
                "delta_AP_small": row["delta_AP_small_vs_ref"],
                "fps": row["fps_end_to_end"],
                "passes": ok,
            }
        )
        axis = v["axis"].split()[0]
        if ok and (axis not in winners or row["mAP"] > fast[winners[axis]]["mAP"]):
            winners[axis] = v["name"]
    ordered = sorted(winners.values(), key=lambda n: -fast[n]["mAP"])
    # Rule 5: combinations are named by the winners they stack ("members", in greedy order).
    stack, current = [], REF
    steps = []
    for name in ordered:
        candidate_members = stack + [name]
        if len(candidate_members) == 1:
            candidate = name
        else:
            # Several combinations may stack the same members (for example two merge IoUs): take the best.
            match = [
                c["name"]
                for c in combos
                if c.get("members") == candidate_members and c["name"] in fast
            ]
            candidate = max(match, key=lambda n: fast[n]["mAP"]) if match else None
        if candidate is None or candidate not in fast:
            steps.append(
                {
                    "add": name,
                    "candidate": candidate,
                    "kept": None,
                    "reason": "combination not evaluated",
                }
            )
            continue
        gain = fast[candidate]["mAP"] - fast[current]["mAP"]
        fps = fast[candidate]["fps_end_to_end"]
        keep = (len(candidate_members) == 1 or gain >= STACK_MARGIN) and (
            fps is None or fps >= MIN_FPS
        )
        steps.append(
            {
                "add": name,
                "candidate": candidate,
                "gain_vs_stack": r4(gain),
                "fps": fps,
                "kept": keep,
            }
        )
        if keep:
            stack, current = candidate_members, candidate
    veto = None
    if current != REF and current in full and REF in full:
        veto = full[current]["mAP"] < full[REF]["mAP"]
    selected = REF if veto else current
    return {
        "rule_file": "selection_rule.md (commit 2b658d0, before any CH-13 result)",
        "thresholds": {
            "margin": MARGIN,
            "ap_small_tolerance": AP_SMALL_TOLERANCE,
            "ensemble_margin": ENSEMBLE_MARGIN,
            "stack_margin": STACK_MARGIN,
            "min_fps": MIN_FPS,
        },
        "screened_on": "val_fast",
        "screening": screened,
        "axis_winners": winners,
        "greedy_steps": steps,
        "stack": stack,
        "candidate_before_veto": current,
        "val_full_veto": veto,
        "selected": selected,
        "val_full": {
            n: {k: full[n][k] for k in ("mAP", "AP50", "AP75", "AP_small")}
            for n in full
        },
    }


def per_class_rows(singles, combos) -> list[dict]:
    out = []
    for split in ("val_fast", "val_full"):
        for v in singles + combos:
            r = run(split, v["name"])
            if r is None:
                continue
            for c in r["metrics"]["per_class"]:
                out.append(
                    {
                        "split": split,
                        "variant": v["name"],
                        "class": c["name"],
                        "num_gt": c["num_gt"],
                        "AP": r4(c["AP"]),
                        "AP50": r4(c["AP50"]),
                    }
                )
    return out


def density_rows(singles, combos) -> list[dict]:
    out = []
    for split in ("val_fast", "val_full"):
        for v in singles + combos:
            r = run(split, v["name"])
            if r is None:
                continue
            for d in r["metrics"]["density"]:
                out.append(
                    {
                        "split": split,
                        "variant": v["name"],
                        "bin": d["bin"],
                        "num_images": d["num_images"],
                        "mAP": r4(d["mAP"]),
                        "AP50": r4(d["AP50"]),
                    }
                )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- figures

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#a3a29c", "#0b0b0b", "#52514e"


def figures(rows, selection) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = HERE / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fast = [r for r in rows if r["split"] == "val_fast" and r["variant"] != REF]
    passed = {s["variant"] for s in selection["screening"] if s["passes"]}
    passed |= {s["candidate"] for s in selection["greedy_steps"] if s.get("kept")}

    # 1. val_fast gain of every variant over the reference.
    fast = sorted(fast, key=lambda r: r["delta_mAP_vs_ref"])
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(fast) + 1.4))
    y = range(len(fast))
    ax.barh(
        list(y),
        [r["delta_mAP_vs_ref"] for r in fast],
        color=[
            ORANGE
            if r["variant"] == selection["selected"]
            else BLUE
            if r["variant"] in passed
            else GRAY
            for r in fast
        ],
        height=0.7,
        edgecolor="white",
        linewidth=1,
    )
    ax.set_yticks(
        list(y), [f"{r['variant']} ({r['axis'].split()[0]})" for r in fast], fontsize=8
    )
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(MARGIN, color=MUTED, lw=0.8, ls="--")
    ax.text(
        MARGIN, len(fast) - 0.4, " +0.003 margin", color=MUTED, fontsize=7, va="bottom"
    )
    ax.set_xlabel(
        "val_fast mAP minus reference (orange: selected, blue: passes the rule, grey: does not)"
    )
    ax.set_title(
        f"CH-13 inference variants vs reference (val_fast, R = {selection_ref(rows, 'val_fast'):.4f})",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "val_fast_deltas.png", dpi=130)
    plt.close(fig)

    # 2. Accuracy vs end-to-end speed (variants with a speed measurement).
    timed = [r for r in rows if r["split"] == "val_fast" and r["fps_end_to_end"]]
    timed.sort(
        key=lambda r: r["variant"] == selection["selected"]
    )  # draw the selected point last
    below = {REF, "combo_flip_tile_nms06"}  # labels that would overlap a neighbour
    if timed:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for r in timed:
            is_sel = r["variant"] == selection["selected"]
            ax.scatter(
                r["fps_end_to_end"],
                r["mAP"],
                s=48,
                color=ORANGE if is_sel else BLUE,
                edgecolor="white",
                linewidth=1.5,
                zorder=3,
            )
            ax.annotate(
                r["variant"],
                (r["fps_end_to_end"], r["mAP"]),
                textcoords="offset points",
                xytext=(5, -10 if r["variant"] in below else 4),
                fontsize=7,
                color=INK,
            )
        ax.axvline(MIN_FPS, color=MUTED, lw=0.8, ls="--")
        ax.text(
            MIN_FPS,
            ax.get_ylim()[0],
            " 1.6 FPS limit",
            color=MUTED,
            fontsize=7,
            va="bottom",
        )
        ax.set_xscale("log")
        ax.set_xlabel("end-to-end FPS, RTX 5070 Laptop, batch 1 (log scale)")
        ax.set_ylabel("val_fast mAP")
        ax.set_title("Accuracy vs speed (orange: selected)", fontsize=10)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / "accuracy_vs_speed.png", dpi=130)
        plt.close(fig)

    # 3. val_full reference vs selected, per class and per density bin.
    sel = selection["selected"]
    ref_run, sel_run = run("val_full", REF), run("val_full", sel)
    if ref_run and sel_run and sel != REF:
        fig, axes = plt.subplots(
            1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2, 1]}
        )
        groups = [
            (
                [c["name"] for c in ref_run["metrics"]["per_class"]],
                "per_class",
                "AP",
                "class",
            ),
            (
                [d["bin"] for d in ref_run["metrics"]["density"]],
                "density",
                "mAP",
                "targets per frame",
            ),
        ]
        for ax, (labels, key, metric, xlabel) in zip(axes, groups):
            x = range(len(labels))
            a = [row[metric] for row in ref_run["metrics"][key]]
            b = [row[metric] for row in sel_run["metrics"][key]]
            ax.bar(
                [i - 0.2 for i in x],
                a,
                width=0.38,
                color=BLUE,
                label=f"{REF} (CH-12 baseline)",
                edgecolor="white",
                linewidth=1,
            )
            ax.bar(
                [i + 0.2 for i in x],
                b,
                width=0.38,
                color=ORANGE,
                label=f"{sel} (selected)",
                edgecolor="white",
                linewidth=1,
            )
            ax.set_xticks(
                list(x), labels, fontsize=8, rotation=30 if key == "per_class" else 0
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"val_full {metric}")
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle("val_full: reference vs selected inference", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "val_full_class_density.png", dpi=130)
        plt.close(fig)


def selection_ref(rows, split) -> float:
    return next(r["mAP"] for r in rows if r["split"] == split and r["variant"] == REF)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    singles, combos = variants()
    rows = comparison_rows(singles, combos)
    write_csv(HERE / "comparison.csv", rows)
    write_csv(HERE / "per_class.csv", per_class_rows(singles, combos))
    write_csv(HERE / "density.csv", density_rows(singles, combos))
    selection = select(rows, singles, combos)
    (HERE / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    if not args.tables_only:
        figures(rows, selection)
    print(
        f"selected: {selection['selected']} (stack {selection['stack']}, veto {selection['val_full_veto']})"
    )


if __name__ == "__main__":
    main()
