"""OPT-04: stop_after_epoch run-config key, the P/FB configs, the 1600 pipeline copy, the inference wrapper and gate helpers."""

import json
import subprocess
import sys

import pytest

from movd import data, ultra

ROOT = data.REPO_ROOT
RUNS = ROOT / "configs" / "runs"
sys.path.insert(0, str(ROOT / "scripts"))

import opt04_gates  # noqa: E402
import opt04_infer  # noqa: E402


def read(name: str) -> dict:
    return json.loads((RUNS / name).read_text(encoding="utf-8"))


def test_stop_after_epoch_valid_in_full_run():
    config = ultra.load_run_config(RUNS / "opt04_rtdetrx_e1.json")
    assert config["stop_after_epoch"] == 1 and config["epochs"] == 6


@pytest.mark.parametrize("value", [0, 6, 7, -1, True, 1.5])
def test_stop_after_epoch_rejects_bad_values(value):
    config = {**read("opt04_rtdetrx_e1.json"), "stop_after_epoch": value}
    with pytest.raises(ValueError):
        ultra.validate_run_config(config)


def test_stop_after_epoch_not_passed_to_ultralytics():
    kwargs = ultra.train_kwargs(
        read("opt04_rtdetrx_e1.json"), "d.yaml", ROOT / "outputs" / "x", "0"
    )
    assert "stop_after_epoch" not in kwargs


def test_primary_differs_from_model_a_only_in_weights():
    a, p = read("final_rtdetr.json"), read("opt04_rtdetrx_e1.json")
    changed = {k for k in a.keys() | p.keys() if a.get(k) != p.get(k)}
    assert changed == {"run_id", "weights", "note", "stop_after_epoch"}
    assert p["weights"] == "outputs/weights/rtdetr-x.pt"
    project = ROOT / "outputs" / "x"
    assert ultra.train_kwargs(a, "d.yaml", project, "0") == ultra.train_kwargs(
        p, "d.yaml", project, "0"
    )


def test_fallback_differs_from_model_a_only_in_imgsz():
    a, fb = read("final_rtdetr.json"), read("opt04_rtdetrl1600_e1.json")
    changed = {k for k in a.keys() | fb.keys() if a.get(k) != fb.get(k)}
    assert changed == {"run_id", "note", "overrides", "stop_after_epoch"}
    assert fb["weights"] == a["weights"] and fb["imgsz"] == 1344
    project = ROOT / "outputs" / "x"
    ka = ultra.train_kwargs(a, "d.yaml", project, "0")
    kf = ultra.train_kwargs(ultra.validate_run_config(fb), "d.yaml", project, "0")
    assert {k for k in ka if ka[k] != kf[k]} == {"imgsz"} and kf["imgsz"] == 1600


@pytest.mark.parametrize(
    ("smoke", "run"),
    [
        ("opt04_rtdetrx_smoke.json", "opt04_rtdetrx_e1.json"),
        ("opt04_rtdetrl1600_smoke.json", "opt04_rtdetrl1600_e1.json"),
    ],
)
def test_smoke_configs_match_their_run(smoke, run):
    s, r = ultra.validate_run_config(read(smoke)), read(run)
    assert s["weights"] == r["weights"] and s["overrides"] == r["overrides"]
    assert (
        s["epochs"] == 2
        and s["smoke_fraction"] == 0.0165
        and "stop_after_epoch" not in s
    )


def test_inference_1600_changes_only_pass_sizes():
    base, new = read("final_inference.json"), read("opt04_inference_1600.json")
    assert [p["imgsz"] for p in new["config"]["passes"]] == [1600, 1600, 1600]
    for p in base["config"]["passes"]:
        p["imgsz"] = 1600
    assert base == new


def test_single_pass_follows_pipeline_size():
    pipeline = read("opt04_inference_1600.json")["config"]
    config = opt04_infer.single_pass(pipeline)
    assert config["passes"] == [{"model": "baseline", "imgsz": 1600, "views": "full"}]
    assert (
        opt04_infer.single_pass(read("final_inference.json")["config"])["passes"][0][
            "imgsz"
        ]
        == 1344
    )


def test_parse_model_spec():
    name, weights, pipeline = opt04_infer.parse_model_spec(
        "FB=out/w.pt@configs/runs/opt04_inference_1600.json"
    )
    assert (name, weights.as_posix(), pipeline.name) == (
        "FB",
        "out/w.pt",
        "opt04_inference_1600.json",
    )
    assert opt04_infer.parse_model_spec("A=w.pt")[2] == opt04_infer.DEFAULT_PIPELINE


def test_gate_check_and_secondary():
    assert opt04_gates.check("m", 0.2922, 0.2922)["pass"]
    assert not opt04_gates.check("m", 0.29219, 0.2922)["pass"]
    assert (
        opt04_gates.check("t", 90.0, 90.0, "<=")["pass"]
        and not opt04_gates.check("t", None, 1)["pass"]
    )

    def result(v, b):
        return {
            "metrics": dict.fromkeys(opt04_gates.SECONDARY, v),
            "density": [
                {"bin": k, "mAP": b} for k in ("0-19", "20-49", "50-99", ">=100")
            ],
        }

    rows = opt04_gates.secondary(result(0.3, 0.2), result(0.2, 0.2))
    assert len(rows) == 9 and sum(r["lead"] for r in rows) == 5  # ties are not leads


def test_log_problems(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "epoch 1 loss 1.0\ntorch.OutOfMemoryError: CUDA out of memory\nReducing batch size to 0\n"
    )
    assert len(opt04_gates.log_problems([log])) == 2
    log.write_text("epoch 1/6 giou_loss 0.7\n")
    assert opt04_gates.log_problems([log]) == []


def test_thresholds_match_the_rule():
    assert opt04_gates.G2 == {"mAP": 0.2922, "AP_small": 0.1397, "bin_ge100": 0.3465}
    assert opt04_gates.G3 == {
        "mAP": 0.2890,
        "AP_small": 0.1474,
        "bin_ge100": 0.3447,
        "secondary_wins": 5,
    }
    assert opt04_gates.G5_MAP == 0.2923 and opt04_gates.SPREAD == 0.0063
    rule = (ROOT / "results" / "opt-04-model-scale" / "selection_rule.md").read_text(
        encoding="utf-8"
    )
    for value in (
        "0.2922",
        "0.1397",
        "0.3465",
        "0.2890",
        "0.1474",
        "0.3447",
        "0.2923",
        "7.0 GB",
        "Thu 12 Nov 18:00",
    ):
        assert value in rule


def test_stop_run_refuses_resume():
    proc = subprocess.run(
        [sys.executable, "scripts/train_ultra.py", "--config", "configs/runs/opt04_rtdetrx_e1.json",
         "--device", "cpu", "--resume"],
        check=False, cwd=ROOT, capture_output=True, text=True,
    )  # fmt: skip
    assert proc.returncode == 2 and "not resumed" in proc.stderr
