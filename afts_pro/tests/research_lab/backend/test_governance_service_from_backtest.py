from datetime import datetime, timezone
from pathlib import Path

from research_lab.backend.core.backtests.models import BacktestResult, BacktestKpiSummary, BacktestEngineDetail
from research_lab.backend.core.backtests.persistence import BacktestPersistence
from research_lab.backend.core.experiments.persistence import ExperimentPersistence
from research_lab.backend.core.governance.models import ModelStage, ModelType
from research_lab.backend.core.governance.registry import GovernanceRegistry
from research_lab.backend.core.governance.service import GovernanceService
from research_lab.backend.core.rl.service import RLService
from research_lab.backend.core.rl.runner import RLRunner
from research_lab.backend.core.rl.reward_verifier import RLRewardVerifier
from research_lab.backend.core.rl_experiments.persistence import RlExperimentPersistence
from research_lab.backend.core.job_runner import InMemoryJobRunner


def _service(tmp_path: Path) -> tuple[GovernanceService, str]:
    backtest_dir = tmp_path / "backtests"
    backtest_persistence = BacktestPersistence(backtest_dir)
    experiment_persistence = ExperimentPersistence(tmp_path / "experiments")
    rl_service = RLService(job_runner=InMemoryJobRunner(), rl_runner=RLRunner(policies_dir=tmp_path / "policies", verifier=RLRewardVerifier()))
    rl_exp_persistence = RlExperimentPersistence(tmp_path / "rl_experiments")
    registry = GovernanceRegistry(tmp_path / "governance")
    return (
        GovernanceService(
            registry=registry,
            backtest_persistence=backtest_persistence,
            experiment_persistence=experiment_persistence,
            rl_service=rl_service,
            rl_experiment_persistence=rl_exp_persistence,
        ),
        backtest_dir,
    )


def test_register_and_promote_backtest(tmp_path: Path) -> None:
    service, backtest_dir = _service(tmp_path)
    backtest_persistence = service.backtest_persistence
    backtest_result = BacktestResult(
        id="bt1",
        created_at=datetime.now(timezone.utc),
        mode="graph",
        kpi_summary=BacktestKpiSummary(
            total_return=0.5,
            mean_return=0.05,
            std_return=0.01,
            profit_factor=1.8,
            win_rate=0.6,
            max_drawdown=0.1,
            trade_count=10,
        ),
        engine_detail=BacktestEngineDetail(window_kpis=[]),
        metadata={},
        ftmo_risk_summary={
            "passed": True,
            "first_breach": None,
            "worst_daily_drawdown_pct": 0.02,
            "worst_total_drawdown_pct": 0.05,
            "config": {},
        },
    )
    backtest_persistence.save_result(backtest_result)

    entry = service.register_from_backtest(name="model-bt", backtest_id="bt1")
    assert entry.type == ModelType.BACKTEST_STRATEGY
    assert entry.stage == ModelStage.CANDIDATE
    assert entry.kpi.total_return == 0.5
    assert entry.ftmo.passed is True

    promoted = service.promote(entry.id, ModelStage.QUALIFIED, note="looks good")
    assert promoted.stage == ModelStage.QUALIFIED
