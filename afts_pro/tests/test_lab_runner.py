import json
from pathlib import Path
from typing import Any, Dict

from afts_pro.lab.models import LabExperiment, LabSweepDefinition, RunResult
from afts_pro.lab.runner import LabRunner


class FakeSimApi:
    def __init__(self, base_dir: Path, metrics: Dict[str, Any] | None = None) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics or {"pf": 1.0, "winrate": 0.5, "mdd": 0.1, "trades": 5}
        self._counter = 0

    def run_backtest(self, profile_name: str, overrides: Dict[str, Any] | None = None, seed: int | None = None) -> RunResult:
        run_id = f"run_{self._counter}"
        self._counter += 1
        run_path = self.base_dir / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        metrics_path = run_path / "metrics.json"
        metrics_path.write_text(json.dumps(self.metrics))
        return RunResult(run_id=run_id, run_path=str(run_path), metrics=self.metrics)


def _lab_config(tmp_path: Path) -> Dict[str, Any]:
    return {
        "default_mode": "strategy_backtest",
        "base_profile": "sim",
        "metrics": ["pf", "winrate", "mdd", "trades"],
        "output": {"root_dir": str(tmp_path / "runs" / "lab"), "save_kpi_matrix": True, "save_equity": False, "save_trades": False},
    }


def test_single_experiment_run(tmp_path):
    sim_api = FakeSimApi(tmp_path / "runs" / "raw")
    cfg = _lab_config(tmp_path)
    runner = LabRunner(cfg, sim_api)
    exp = LabExperiment(id="exp1", name="test", mode="strategy_backtest", base_profile="sim", params={"orb.box": 10}, seed=123, meta={})
    result = runner.run_experiment(exp)
    assert result.params == exp.params
    assert result.metrics
    exp_dir = Path(cfg["output"]["root_dir"]) / "single" / "experiments" / exp.id
    assert exp_dir.exists()
    assert (exp_dir / "experiment_config.yaml").exists()
    assert (exp_dir / "metrics.json").exists()


def test_simple_grid_sweep(tmp_path):
    sim_api = FakeSimApi(tmp_path / "runs" / "raw")
    cfg = _lab_config(tmp_path)
    runner = LabRunner(cfg, sim_api)
    sweep_def = LabSweepDefinition(id="sweep1", type="grid", params={"a": [1, 2], "b": ["x", "y"]}, max_experiments=None, seed=None)
    results = runner.run_sweep(sweep_def)
    assert len(results) == 4
    kpi_path = Path(cfg["output"]["root_dir"]) / sweep_def.id / "kpis.parquet"
    assert kpi_path.exists()


def test_determinism_with_seed(tmp_path):
    metrics = {"pf": 2.0, "winrate": 0.6, "mdd": 0.2, "trades": 7}
    sim_api1 = FakeSimApi(tmp_path / "runs" / "raw1", metrics=metrics)
    sim_api2 = FakeSimApi(tmp_path / "runs" / "raw2", metrics=metrics)
    cfg = _lab_config(tmp_path)
    runner1 = LabRunner(cfg, sim_api1)
    runner2 = LabRunner(cfg, sim_api2)
    sweep_def = LabSweepDefinition(id="sweep2", type="grid", params={"x": [1, 2]}, max_experiments=None, seed=42)
    res1 = runner1.run_sweep(sweep_def)
    res2 = runner2.run_sweep(sweep_def)
    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2):
        assert r1.metrics == r2.metrics
        assert r1.params == r2.params
