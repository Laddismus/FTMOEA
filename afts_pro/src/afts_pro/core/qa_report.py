from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import yaml

from afts_pro.core.e2e_runner import E2ESimConfig, run_e2e_sim
from afts_pro.core.qa_config import QAConfig
from afts_pro.core.train_controller import TrainController, TrainJobConfig
from afts_pro.lab.runner import LabRunner
from afts_pro.lab.models import LabExperiment, LabSweepDefinition, RunResult
from afts_pro.analysis.quant_analyzer import QuantAnalyzer
from afts_pro.analysis.models import QuantConfig

logger = logging.getLogger(__name__)


@dataclass
class QACheckResult:
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QASectionResult:
    name: str
    checks: List[QACheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


@dataclass
class QAReport:
    sections: List[QASectionResult]
    generated_at: datetime
    commit_ref: Optional[str] = None
    notes: Optional[str] = None

    @property
    def all_passed(self) -> bool:
        return all(s.passed for s in self.sections)


def run_e2e_section() -> QASectionResult:
    cfg_path = "configs/modes/sim_e2e_acceptance.yaml"
    result = run_e2e_sim(E2ESimConfig(config_path=cfg_path))
    checks = [
        QACheckResult(
            name="e2e_sim_run_completed",
            passed=True,
            details={
                "num_trades": result.num_trades,
                "equity_start": result.equity_start,
                "equity_end": result.equity_end,
                "run_dir": str(result.run_dir),
            },
        ),
        QACheckResult(name="e2e_has_trades", passed=result.num_trades > 0, details={"num_trades": result.num_trades}),
        QACheckResult(name="e2e_has_rl_signals", passed=result.has_rl_signals, details={"has_rl_signals": result.has_rl_signals}),
    ]
    return QASectionResult(name="e2e_sim_rl", checks=checks)


def run_train_smoke_section(tmp_root: Optional[Path] = None) -> QASectionResult:
    tmp_root = tmp_root or Path("runs/qa_train_smoke")
    tmp_root.mkdir(parents=True, exist_ok=True)
    env_cfg = tmp_root / "env.yaml"
    env_cfg.write_text("env_type: risk\nobservation: {}\n")
    agent_cfg = tmp_root / "agent.yaml"
    agent_cfg.write_text("action_mode: continuous\n")
    job = TrainJobConfig(agent_type="risk", env_config_path=str(env_cfg), agent_config_path=str(agent_cfg), output_dir=str(tmp_root / "out"))
    controller = TrainController()
    summary = controller.run_train_job(job)
    checks = [
        QACheckResult(name="train_run_completed", passed=True, details={"episodes": summary.episodes, "mean_return": summary.mean_return}),
        QACheckResult(name="train_output_dir", passed=Path(summary.output_dir).exists(), details={"output_dir": summary.output_dir}),
    ]
    return QASectionResult(name="train_smoke", checks=checks)


class _FakeSimApi:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def run_backtest(self, profile_name: str, overrides=None, seed=None) -> RunResult:
        run_id = f"run_{self.counter}"
        self.counter += 1
        run_path = self.base_dir / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        metrics = {"pf": 1.0}
        (run_path / "metrics.json").write_text(json.dumps(metrics))
        return RunResult(run_id=run_id, run_path=str(run_path), metrics=metrics)


def run_lab_smoke_section(tmp_root: Optional[Path] = None) -> QASectionResult:
    tmp_root = tmp_root or Path("runs/qa_lab_smoke")
    cfg = {"output": {"root_dir": str(tmp_root / "runs")}, "metrics": ["pf"], "output_root": str(tmp_root / "runs")}
    sim_api = _FakeSimApi(tmp_root / "raw")
    runner = LabRunner(cfg, sim_api)
    exp = LabExperiment(id="qa", name="qa", mode="strategy_backtest", base_profile="sim", params={}, seed=None, meta={})
    res = runner.run_experiment(exp)
    checks = [
        QACheckResult(name="lab_experiment_completed", passed=True, details={"run_id": res.run_id}),
        QACheckResult(name="lab_metrics_written", passed=(Path(cfg["output"]["root_dir"]) / "single" / "experiments" / exp.id / "metrics.json").exists()),
    ]
    return QASectionResult(name="lab_smoke", checks=checks)


def run_quant_smoke_section(tmp_root: Optional[Path] = None) -> QASectionResult:
    tmp_root = tmp_root or Path("runs/qa_quant_smoke")
    cfg = QuantConfig(
        rolling={"window_bars": 10, "step_bars": 5, "metrics": ["pf", "winrate", "mdd", "avg_r", "volatility"]},
        monte_carlo={"enabled": True, "n_scenarios": 5, "horizon_trades": 5, "sampling": "bootstrap"},
        drift={"enabled": True, "threshold_std": 2.5},
        regimes={"enabled": True, "n_regimes": 3, "window": 5},
        output={"root_dir": str(tmp_root), "save_rolling_kpis": False, "save_monte_carlo": False, "save_drift": False, "save_regimes": False},
    )
    qa = QuantAnalyzer(cfg)
    equity = pd.DataFrame({"equity": np.linspace(100, 110, 30)})
    trades = pd.DataFrame({"pnl": np.random.normal(0.1, 0.1, size=30), "r_multiple": np.random.normal(0.1, 0.1, size=30)})
    rolling = qa.rolling_kpis(equity, trades)
    checks = [
        QACheckResult(name="quant_rolling_non_empty", passed=not rolling.df.empty),
    ]
    return QASectionResult(name="quant_smoke", checks=checks)


def run_pytest_smoke_section(pytest_args: Optional[List[str]] = None) -> QASectionResult:
    try:
        import pytest
    except ImportError:  # pragma: no cover - optional
        return QASectionResult(name="pytest_smoke", checks=[QACheckResult(name="pytest_not_available", passed=False)])
    args = pytest_args or ["-q", "tests"]
    code = pytest.main(args)
    checks = [QACheckResult(name="pytest_exit_code", passed=code == 0, details={"exit_code": code})]
    return QASectionResult(name="pytest_smoke", checks=checks)


def run_qa_suite(config: Optional[QAConfig] = None) -> QAReport:
    config = config or QAConfig()
    sections: List[QASectionResult] = []
    if config.enable_e2e:
        sections.append(run_e2e_section())
    if config.enable_train_smoke:
        sections.append(run_train_smoke_section())
    if config.enable_lab_smoke:
        sections.append(run_lab_smoke_section())
    if config.enable_quant_smoke:
        sections.append(run_quant_smoke_section())
    if config.enable_pytest_smoke:
        sections.append(run_pytest_smoke_section(config.pytest_args))
    return QAReport(sections=sections, generated_at=datetime.utcnow())


def save_report(report: QAReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = report.generated_at.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"qa_report_{ts}.json"
    txt_path = output_dir / f"qa_report_{ts}.txt"
    json_path.write_text(
        json.dumps(
            {
                "generated_at": report.generated_at.isoformat(),
                "all_passed": report.all_passed,
                "sections": [
                    {
                        "name": s.name,
                        "passed": s.passed,
                        "checks": [{"name": c.name, "passed": c.passed, "details": c.details} for c in s.checks],
                    }
                    for s in report.sections
                ],
            },
            indent=2,
        )
    )
    lines = ["AFTS-PRO QA REPORT", "==================", f"Generated at: {report.generated_at.isoformat()}", ""]
    for sec in report.sections:
        status = "PASS" if sec.passed else "FAIL"
        lines.append(f"Section: {sec.name} .... {status}")
        for check in sec.checks:
            c_status = "PASS" if check.passed else "FAIL"
            lines.append(f"  - {check.name} ... {c_status} ({check.details})")
        lines.append("")
    lines.append(f"OVERALL STATUS: {'PASSED' if report.all_passed else 'FAILED'}")
    txt_path.write_text("\n".join(lines))
    return json_path, txt_path
