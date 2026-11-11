"""Budget accounting for the separately authorized OPT-10 c0 continuation."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import opt10_c0_continue as continuation


@pytest.fixture
def accounting(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, "EVID", tmp_path)
    (tmp_path / "contract.json").write_text(
        json.dumps(
            {
                "chunk": "10.3c",
                "gpu_budget_minutes": 25,
                "gpu_job_estimate_minutes": 13,
                "gpu_job": "cache_c0_confirmation",
            }
        )
    )
    return tmp_path


def test_continuation_uses_its_own_log_without_consuming_prior_phase_budget(accounting):
    with continuation.gpu_job():
        pass
    record = json.loads((accounting / "gpu_log.json").read_text())
    assert record["budget_minutes"] == 25 and record["chunk"] == "10.3c"
    assert len(record["jobs"]) == 1 and record["jobs"][0]["status"] == "ok"


def test_refused_reservation_does_not_modify_the_existing_log(accounting):
    path = accounting / "gpu_log.json"
    path.write_text(json.dumps({"jobs": [{"seconds": 13 * 60}]}))
    before = path.read_bytes()
    with (
        pytest.raises(SystemExit, match="continuation budget"),
        continuation.gpu_job(),
    ):
        pytest.fail("refused job must not start")
    assert path.read_bytes() == before


def test_failed_job_is_accounted_as_failed(accounting):
    with pytest.raises(RuntimeError), continuation.gpu_job():
        raise RuntimeError("test failure")
    record = json.loads((accounting / "gpu_log.json").read_text())
    assert record["jobs"][0]["status"] == "failed"
