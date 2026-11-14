"""OPT-11: the N-way soup average equals OPT-08's two-way soup arithmetic."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import opt08_soup as s8
import opt11_night as o11


def _states():
    g = torch.Generator().manual_seed(0)
    a = {
        "w": torch.randn(4, 3, generator=g),
        "h": torch.randn(5, generator=g).half(),
        "n": torch.tensor([3]),
    }
    b = {
        "w": torch.randn(4, 3, generator=g),
        "h": torch.randn(5, generator=g).half(),
        "n": torch.tensor([7]),
    }
    return a, b


def test_two_way_equals_opt08_soup_state():
    a, b = _states()
    for w in (0.75, 0.8):
        ref = s8.soup_state(a, b, w)
        got = o11.nway_state([a, b], [w, 1.0 - w])
        assert all(torch.equal(ref[k], got[k]) for k in ref)


def test_non_float_from_first_and_weights_sum_to_one():
    a, b = _states()
    got = o11.nway_state([a, b, a], [0.5, 0.25, 0.25])
    assert torch.equal(got["n"], a["n"])
    for parts in o11.SOUPS.values():
        assert abs(sum(w for _, w in parts) - 1.0) < 1e-9
