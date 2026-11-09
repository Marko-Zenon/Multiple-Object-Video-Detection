from pathlib import Path
import json, re, subprocess, datetime, zoneinfo, csv

R = Path.cwd()
s = Path("report.md").read_text()


def j(p):
    return json.loads(Path(p).read_text())


trace = j("docs/report-support/metric_trace.json")
subprocess.run(
    ["git", "merge-base", "--is-ancestor", trace["evidence_commit"], "HEAD"], check=True
)
for t in trace["metric_claims"]:
    d = j(t["source"])
    for k in t["pointer"].split("."):
        d = d[k]
    assert f"{d:.4f}" == t["display"]
    assert t["display"] in s
# Reuse the previous CH-15 report/link validator's numerical, metadata, image,
# table and per-class checks; its old task-specific board-edit policy is inapplicable.
# Equivalent checks below are self-contained; the original session helper is
# also run separately, excluding its obsolete CH-15 board-edit assertion.
# New tables: check individual cells (not only presence elsewhere in the report).
keys = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
for line in s.splitlines():
    m = re.match(r"\| \[([^]]+)\]\(results/([^/]+)/metrics.json\) \| (.*)", line)
    if not m:
        continue
    d = j(f"results/{m[2]}/metrics.json")
    v = d.get("val_full", d)
    assert v["num_images"] == 2846 and v["max_dets"] == 500
    values = [c.strip() for c in m[3].strip("|").split("|")]
    assert values[:6] == [f"{v['metrics'][k]:.4f}" for k in keys], m[1]
# Validate all numbers in retained Faster R-CNN metric/per-class tables regardless
# of the presence of the historical local helper.
pcs = []
for run in ["frcnn_baseline", "frcnn_rpn_pre3000", "frcnn_roi_pos50"]:
    d = j(f"results/{run}/metrics.json")["val_full"]
    pcs.append(d["per_class"])
    speed = j(f"results/{run}/speed.json")["inference"]
    line = next(l for l in s.splitlines() if l.startswith("| [" + run + "]"))
    assert line.endswith(f"{speed['fps']:.2f} | {speed['latency_ms_per_image']:.2f} |")
for items in zip(*pcs):
    assert (
        "| "
        + items[0]["name"]
        + " | "
        + " | ".join(f"{x['AP']:.4f}" for x in items)
        + " |"
        in s
    )
eda = j("results/ch-06-eda/key_numbers.json")
assert [(x["clips"], x["frames"]) for x in eda["split_overview"]] == [
    (56, 24201),
    (7, 2846),
    (48, 3811),
]
assert [
    (x["target objects"], x["unique ignore regions"]) for x in eda["split_overview"][:2]
] == [(1105516, 63843), (114132, 3995)]
assert eda["class_targets"][2]["train %"] == 45.71
assert eda["class_targets"][0]["val %"] == 44.08
assert eda["class_targets"][6]["val targets"] == 264
assert eda["size_by_split"][0]["small < 32 px %"] == 24.91
assert eda["size_after_resize"][4]["< 16 px %"] == 8.26
assert eda["redundancy"][0]["targets of t+k already in t %"] == 99.5
assert eda["subsampling"]["tracks seen in train_rubric %"] == 98.6
assert eda["subsampling"]["gap median"] == 8
protocol = j("configs/protocol.json")
assert protocol["seed"] == 42 and protocol["rubric_budget"]["epochs"] == 6
assert (
    protocol["rubric_budget"]["images_seen"] == 16800
    and protocol["rubric_budget"]["input_long_side"] == 1344
)
assert protocol["splits"]["train_rubric"]["frames_per_clip"] == 50
assert (
    protocol["splits"]["train_rubric"]["num_images"] == 2800
    and protocol["splits"]["val_fast"]["num_images"] == 714
)
sanity = j("results/ch-03-evaluator/sanity.json")["checks"]
assert sanity["gt_as_predictions_val_full"]["metrics"]["mAP"] == 1
assert sanity["gt_plus_ignore_region_dets_val_fast"]["metrics"]["mAP"] == 1
assert f"{sanity['control_crowd_removed_val_fast']['metrics']['mAP']:.4f}" == "0.8667"
assert sanity["label_shift_val_fast"]["metrics"]["mAP"] < 0.001
for run in [
    "frcnn_baseline",
    "frcnn_rpn_pre3000",
    "frcnn_roi_pos50",
    "retina_baseline",
    "retina_anchors",
    "retina_focal_gamma15",
]:
    c = j(f"results/{run}/config.json")
    assert c["batch_size"] == 2 and c["epochs"] == 6
for run in ["yolo_baseline_marko_3050", "rtdetr_baseline"]:
    c = j(f"results/{run}/config.json")["ultralytics_train_kwargs"]
    assert c["batch"] == 1 and c["epochs"] == 6 and c["imgsz"] == 1344
assert j("results/retina_baseline/config.json")["focal_gamma"] == 2
assert j("results/retina_focal_gamma15/config.json")["focal_gamma"] == 1.5
assert j("results/retina_focal_gamma15/config.json")["focal_alpha"] == 0.25
rows = list(csv.DictReader(Path("results/ch-11-rtdetr/curves.csv").open()))
for e in range(1, 7):
    baseline = next(
        float(x["val_fast_mAP"])
        for x in rows
        if x["run"] == "rtdetr_baseline" and int(x["epoch"]) == e
    )
    cnns = [
        float(x["val_fast_mAP"])
        for x in rows
        if x["run"].startswith(("frcnn_", "retina_", "yolo_")) and int(x["epoch"]) == e
    ]
    assert baseline > max(cnns)
for run, peak in [("rtdetr_baseline", 3), ("yolo_baseline_marko_3050", 2)]:
    assert (
        int(
            max(
                (x for x in rows if x["run"] == run),
                key=lambda x: float(x["val_fast_mAP"]),
            )["epoch"]
        )
        == peak
    )
for run, expected in [
    ("yolo_baseline_marko_3050", "64.21"),
    ("rtdetr_baseline", "10.60"),
]:
    d = j(f"results/ch-11-rtdetr/speed_recheck/{run}.json")
    assert f"{d['fps']:.2f}" == expected
assert "36.37 FPS / 27.49 ms per image on RTX 3050 Laptop GPU" in s
speed = j("results/yolo_baseline_marko_3050/speed.json")
speed = speed.get("inference", speed)
assert (
    f"{speed['fps']:.2f}" == "36.37"
    and f"{speed['latency_ms_per_image']:.2f}" == "27.49"
)
# Directional findings are checked against measured values, not inferred causes.
comparison = list(csv.DictReader(Path("results/ch-11-rtdetr/comparison.csv").open()))
byrun = {x["run"]: x for x in comparison}
for run in ["rtdetr_q500", "rtdetr_lr1e4_wb0"]:
    assert float(byrun[run]["val_full_mAP"]) < float(
        byrun["rtdetr_baseline"]["val_full_mAP"]
    )
assert float(byrun["rtdetr_q500"]["val_full_AP_small"]) < float(
    byrun["rtdetr_baseline"]["val_full_AP_small"]
)
for field in ["val_full_AP_small", "val_full_AP_medium", "val_full_AP_large"]:
    assert float(byrun["rtdetr_baseline"][field]) > max(
        float(x[field]) for x in comparison if x["run"] != "rtdetr_baseline"
    )
density = list(csv.DictReader(Path("results/ch-11-rtdetr/density.csv").open()))
for b in {x["bin"] for x in density}:
    vals = [x for x in density if x["bin"] == b]
    assert max(vals, key=lambda x: float(x["mAP"]))["run"] == "rtdetr_baseline"
submission = list(csv.DictReader(Path("results/submissions.csv").open()))[0]
assert f"{float(submission['val_full_mAP']):.10f}" == "0.1893354654"
assert (
    submission["kaggle_public_score"] == "0.2011538266"
    and submission["kaggle_status"] == "accepted"
)
assert (
    protocol["kaggle"]["public_baseline"] == 0.22163
    and protocol["kaggle"]["final_submissions_selected"] == 2
)
receipt = Path(
    "results/ch-05-submission/frcnn_rpn_pre3000_e6_test/validation.txt"
).read_text()
assert '"data_rows": 3811' in receipt and '"detections": 261940' in receipt
assert "ultralytics==8.4.207" in Path("requirements.txt").read_text()
for run in ["frcnn_baseline", "frcnn_rpn_pre3000", "frcnn_roi_pos50"]:
    h = j(f"results/{run}/hardware.json")
    assert "3.12.14" in str(h) and "2.14.1+cu130" in str(h) and "0.29.1+cu130" in str(h)
# All referenced experiment sources must remain byte-identical to the evidence base.
for dest in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", s):
    if dest.startswith(("results/", "configs/", "movd/", "images/")):
        assert Path(dest).read_bytes() == subprocess.check_output(
            ["git", "show", trace["evidence_commit"] + ":" + dest]
        )

# All report and support-document links; relative Markdown paths resolved per file.
count = 0
for path in [Path("report.md"), *Path("docs/report-support").glob("*.md")]:
    for dest in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text()):
        if dest.startswith(("http:", "https:")):
            continue
        assert (path.parent / dest.split("#")[0]).is_file(), (str(path), dest)
        count += 1
assert set(re.findall(r"^(\w+):", s.split("---")[1], re.M)) == {
    "title",
    "author",
    "course",
    "date",
}
prev = 0
for line in s.splitlines():
    if line.startswith("|"):
        n = line.count("|")
        assert not prev or n == prev
        prev = n
    else:
        prev = 0
assert (
    Path("images/ch-15-ignore-masking-check.png").read_bytes()
    == Path("results/ch-02-protocol/figures/ignore_masking_check.png").read_bytes()
)
from PIL import Image

for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", s):
    Image.open(path).verify()
allowed = {
    "report.md",
    "docs/report-support/deadline.md",
    "docs/report-support/metric_trace.json",
    "docs/report-support/validation.md",
    "docs/report-support/validate_report.py",
}
changed = set(
    subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only"], text=True
    ).splitlines()
)
untracked = set(
    subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], text=True
    ).splitlines()
)
assert (changed | untracked) <= allowed, (changed, untracked)
assert (
    subprocess.check_output(
        [
            "git",
            "diff",
            "HEAD",
            "--",
            "plan.md",
            "homework.ipynb",
            "index.html",
            "assets",
            ".github",
            ".nojekyll",
            "results",
            "configs",
            "movd",
        ],
        text=True,
    )
    == ""
)
dt = datetime.datetime.fromisoformat("2026-11-10T23:55:00+02:00")
assert dt.astimezone(datetime.timezone.utc).isoformat() == "2026-11-10T21:55:00+00:00"
assert (
    dt.astimezone(zoneinfo.ZoneInfo("Europe/Kyiv")).isoformat()
    == "2026-11-10T23:55:00+02:00"
)
subprocess.run(["git", "diff", "--check"], check=True)
print(
    f"PASS: {len(trace['metric_claims'])} traced new metric cells; all retained FRCNN cells; EDA/protocol/speed/curve controls; {count} internal links; images; timezone; documentation-only scope."
)
