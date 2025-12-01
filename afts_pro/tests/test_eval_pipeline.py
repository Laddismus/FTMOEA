import json
from pathlib import Path

from afts_pro.core.benchmark_engine import BenchmarkEngine
from afts_pro.core.eval_controller import EvalConfig, EvalController


def test_benchmark_engine_calculates_scores():
    engine = BenchmarkEngine()
    report = engine.evaluate(
        checkpoint_path="ckpt",
        sim_metrics={"profit_factor": 1.5, "winrate": 0.6, "mdd_pct": 5.0},
        ftmo_metrics={"ftmo_pass": True},
        rl_train_metrics={"reward_slope": 0.1},
    )
    assert report.score != 0
    assert report.kpis["profit_factor"] == 1.5


def test_eval_controller_runs_single_checkpoint(tmp_path):
    ckpt = tmp_path / "ckpt_dir"
    ckpt.mkdir()
    # create fake metrics for benchmark_engine to load
    (ckpt / "metrics.json").write_text(json.dumps({"profit_factor": 1.0, "winrate": 0.5, "mdd_pct": 4.0}))
    (ckpt / "ftmo_metrics.json").write_text(json.dumps({"ftmo_pass": True, "overall_dd": 8.0}))
    (ckpt / "train_summary.json").write_text(json.dumps({"mean_reward": 0.1, "reward_slope": 0.05}))
    cfg = EvalConfig(
        base={},
        rl={"checkpoint_path": str(ckpt)},
        output={"evaluation_root": str(tmp_path), "save_txt": True},
    )
    controller = EvalController(cfg)
    report = controller.run_eval()
    assert Path(report.checkpoint_path) == ckpt
    out_files = list(Path(tmp_path).glob("benchmark_*.json"))
    assert out_files, "benchmark report not written"


def test_multi_eval_selects_best_checkpoint(tmp_path):
    ckpt1 = tmp_path / "ckpt1"
    ckpt2 = tmp_path / "ckpt2"
    ckpt1.mkdir()
    ckpt2.mkdir()
    (ckpt1 / "metrics.json").write_text(json.dumps({"profit_factor": 1.0, "winrate": 0.5, "mdd_pct": 5.0}))
    (ckpt2 / "metrics.json").write_text(json.dumps({"profit_factor": 2.0, "winrate": 0.6, "mdd_pct": 4.0}))
    (ckpt1 / "ftmo_metrics.json").write_text(json.dumps({"ftmo_pass": False}))
    (ckpt2 / "ftmo_metrics.json").write_text(json.dumps({"ftmo_pass": True}))
    (ckpt1 / "train_summary.json").write_text(json.dumps({"reward_slope": 0.0}))
    (ckpt2 / "train_summary.json").write_text(json.dumps({"reward_slope": 0.1}))
    cfg = EvalConfig(base={}, rl={}, output={"evaluation_root": str(tmp_path), "save_txt": False})
    controller = EvalController(cfg)
    comparison = controller.run_multi_eval([str(ckpt1), str(ckpt2)])
    assert comparison.best_checkpoint.endswith("ckpt2")
