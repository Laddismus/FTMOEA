import json
from pathlib import Path

from afts_pro.lab.runner import LabRunner
from afts_pro.lab.models import LabExperiment, LabSweepDefinition, RunResult


class FakeSimApi:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def run_backtest(self, profile_name, overrides=None, seed=None):
        run_id = f"run_{self.counter}"
        self.counter += 1
        run_path = self.base_dir / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        metrics_path = run_path / "metrics.json"
        metrics_path.write_text(json.dumps({"pf": 1.0}))
        return RunResult(run_id=run_id, run_path=str(run_path), metrics={"pf": 1.0})


def test_single_experiment_run_writes_results(tmp_path):
    cfg = {"output": {"root_dir": str(tmp_path / "runs")}}
    sim_api = FakeSimApi(tmp_path / "raw")
    runner = LabRunner(cfg, sim_api)
    exp = LabExperiment(id="exp1", name="test", mode="strategy_backtest", base_profile="sim", params={}, seed=None, meta={})
    res = runner.run_experiment(exp)
    exp_dir = Path(cfg["output"]["root_dir"]) / "single" / "experiments" / exp.id
    assert exp_dir.exists()
    assert (exp_dir / "metrics.json").exists()
    assert res.metrics["pf"] == 1.0


def test_grid_sweep_generates_kpi_matrix(tmp_path):
    cfg = {"output": {"root_dir": str(tmp_path / "runs"), "save_kpi_matrix": True}, "metrics": ["pf"]}
    sim_api = FakeSimApi(tmp_path / "raw")
    runner = LabRunner(cfg, sim_api)
    sweep_def = LabSweepDefinition(id="sweep1", type="grid", params={"a": [1, 2]}, max_experiments=None, seed=None)
    results = runner.run_sweep(sweep_def)
    kpi_path = Path(cfg["output"]["root_dir"]) / sweep_def.id / "kpis.parquet"
    assert kpi_path.exists()
    assert len(results) == 2
