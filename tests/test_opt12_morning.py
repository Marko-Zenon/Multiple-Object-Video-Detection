"""OPT-12 morning dry run on CPU: selection logic, soup building and registry, with a fake single-view scorer.

Uses the smoke run's checkpoints (outputs/opt12_reg_ft_smoke, git-ignored); skipped when they are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "outputs" / "opt12_reg_ft_smoke" / "train" / "weights"
sys.path.insert(0, str(REPO / "scripts"))


def test_lowres_degrade_and_rescale():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    import opt12_morning as m

    img = np.zeros((756, 1344, 3), np.uint8)
    img[100:200, 300:500] = 255
    small = m.degrade(img)
    assert small.shape == (378, 672, 3)
    ys, xs = np.where(small[..., 0] > 127)
    box = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], np.float32)
    back = m.rescale_raw(box, img.shape[1] / small.shape[1], img.shape[0] / small.shape[0])
    assert np.allclose(back, [[300, 100, 500, 200]], atol=2.5)


@pytest.mark.skipif(not (SMOKE / "epoch0.pt").exists(), reason="smoke checkpoints absent")
def test_select_dry_run(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import opt12_morning as m

    from movd import opt12

    monkeypatch.setattr(m, "EVID", tmp_path / "evid")
    monkeypatch.setattr(m, "SEL", tmp_path / "evid" / "selection.json")
    monkeypatch.setattr(m, "REG", tmp_path / "evid" / "registry.json")
    monkeypatch.setattr(m, "SOUP_DIR", tmp_path / "soup")
    monkeypatch.setattr(m, "WEIGHTS_DIR", SMOKE)
    monkeypatch.setattr(opt12, "EVENTS", tmp_path / "events.jsonl")
    monkeypatch.setattr(m, "event", lambda *a, **k: None)
    fake = {"w075": 0.2790}

    def single_view(path):
        p = Path(path).name
        v = fake.get("w075") if "soup_w075" in p else {"epoch0.pt": 0.2700, "epoch1.pt": 0.2760}.get(p)
        if v is None:  # soups: AVG3 / W(alpha)
            v = 0.2800 if "avg3" in p else {"w015n.pt": 0.2805, "w025n.pt": 0.2812, "w050n.pt": 0.2830}[p]
        return {"mAP": v, "metrics": {"AP_small": 0.1}, "predict_seconds": 0.0}

    monkeypatch.setattr(m, "single_view", single_view)
    m.cmd_select(None)
    sel = m.load_json(m.SEL)
    assert sel["done"] and sel["nstar_choice"] == "avg3" and sel["avg3_epochs"] == [1, 2]
    assert sel["alpha_star"] == 0.5 and sel["wstar_screen"] and sel["nstar_screen"]
    reg = m.registry()
    assert reg["wstar"]["path"].endswith("w050n.pt") and reg["nstar"]["path"].endswith("avg3.pt")
    # AVG3 of two epochs equals the float32 mean of their EMA tensors
    a = torch.load(SMOKE / "epoch0.pt", map_location="cpu", weights_only=False)["ema"].state_dict()
    b = torch.load(SMOKE / "epoch1.pt", map_location="cpu", weights_only=False)["ema"].state_dict()
    s = torch.load(reg["avg3"]["path"], map_location="cpu", weights_only=False)
    assert s["optimizer"] is None and s["model"] is None
    k = next(k for k, v in a.items() if v.is_floating_point() and v.numel() > 10)
    assert torch.equal(s["ema"].state_dict()[k], (0.5 * a[k].float() + 0.5 * b[k].float()).to(a[k].dtype))
    # resumable: a second call reuses everything
    m.cmd_select(None)
    assert m.load_json(m.SEL)["nstar_choice"] == "avg3"
