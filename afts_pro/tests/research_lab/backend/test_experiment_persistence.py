from datetime import datetime, timezone
from pathlib import Path

from research_lab.backend.core.backtests.models import BacktestRequest
from research_lab.backend.core.experiments.models import (
    ExperimentConfig,
    ExperimentParamPoint,
    ExperimentRunStatus,
    ExperimentStatus,
    ExperimentStrategyRef,
)
from research_lab.backend.core.experiments.persistence import ExperimentPersistence


def _make_config(exp_id: str) -> ExperimentConfig:
    return ExperimentConfig(
        id=exp_id,
        name="test-exp",
        description=None,
        created_at=datetime.now(timezone.utc),
        strategy=ExperimentStrategyRef(mode="graph"),
        base_backtest=BacktestRequest(mode="graph", returns=[0.1], window=1),
        param_grid=[ExperimentParamPoint(values={"size": 1.0}), ExperimentParamPoint(values={"size": 2.0})],
        tags=["demo"],
        metadata={"note": "test"},
    )


def test_save_and_load_experiment(tmp_path: Path) -> None:
    persistence = ExperimentPersistence(tmp_path)
    config = _make_config("exp1")
    runs = [
        ExperimentRunStatus(run_id="r1", status=ExperimentStatus.PENDING, job_id=None, backtest_id=None),
        ExperimentRunStatus(run_id="r2", status=ExperimentStatus.RUNNING, job_id="job", backtest_id=None),
    ]

    persistence.save_experiment(config, runs)
    loaded = persistence.load_experiment("exp1")

    assert loaded is not None
    cfg, loaded_runs = loaded
    assert cfg.id == config.id
    assert len(loaded_runs) == 2
    assert loaded_runs[1].status == ExperimentStatus.RUNNING


def test_list_experiments(tmp_path: Path) -> None:
    persistence = ExperimentPersistence(tmp_path)
    cfg1 = _make_config("exp1")
    cfg2 = _make_config("exp2")
    runs1 = [ExperimentRunStatus(run_id="r1", status=ExperimentStatus.COMPLETED, job_id=None, backtest_id="b1")]
    runs2 = [
        ExperimentRunStatus(run_id="r2", status=ExperimentStatus.RUNNING, job_id="j2", backtest_id=None),
        ExperimentRunStatus(run_id="r3", status=ExperimentStatus.FAILED, job_id="j3", backtest_id=None),
    ]
    persistence.save_experiment(cfg1, runs1)
    persistence.save_experiment(cfg2, runs2)

    summaries = persistence.list_experiments()

    assert len(summaries) == 2
    status_map = {s.id: s.status for s in summaries}
    assert status_map["exp1"] == ExperimentStatus.COMPLETED
    assert status_map["exp2"] == ExperimentStatus.FAILED
