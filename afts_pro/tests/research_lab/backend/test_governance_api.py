from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from research_lab.backend.app.main import app
from research_lab.backend.app.api import governance as governance_router
from research_lab.backend.core.backtests.models import BacktestResult, BacktestKpiSummary, BacktestEngineDetail
from research_lab.backend.core.backtests.persistence import BacktestPersistence
from research_lab.backend.core.experiments.persistence import ExperimentPersistence
from research_lab.backend.core.governance.registry import GovernanceRegistry
from research_lab.backend.core.governance.service import GovernanceService
from research_lab.backend.core.job_runner import InMemoryJobRunner
from research_lab.backend.core.rl.reward_verifier import RLRewardVerifier
from research_lab.backend.core.rl.runner import RLRunner
from research_lab.backend.core.rl.service import RLService
from research_lab.backend.core.rl_experiments.persistence import RlExperimentPersistence


def setup_router(tmp_path: Path) -> None:
    governance_router._registry = GovernanceRegistry(tmp_path / "gov")
    job_runner = InMemoryJobRunner()
    rl_service = RLService(job_runner=job_runner, rl_runner=RLRunner(policies_dir=tmp_path / "policies", verifier=RLRewardVerifier()))
    governance_router._service = GovernanceService(
        registry=governance_router._registry,
        backtest_persistence=BacktestPersistence(tmp_path / "bt"),
        experiment_persistence=ExperimentPersistence(tmp_path / "exp"),
        rl_service=rl_service,
        rl_experiment_persistence=RlExperimentPersistence(tmp_path / "rlexp"),
    )
    governance_router._rl_service = rl_service


def _persist_backtest(tmp_path: Path) -> str:
    persistence = BacktestPersistence(tmp_path / "bt")
    result = BacktestResult(
        id="bt-api",
        created_at=datetime.now(timezone.utc),
        mode="graph",
        kpi_summary=BacktestKpiSummary(
            total_return=0.4,
            mean_return=0.04,
            std_return=0.01,
            profit_factor=1.5,
            win_rate=0.55,
            max_drawdown=0.1,
            trade_count=8,
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
    persistence.save_result(result)
    return "bt-api"


def test_governance_api_flow(tmp_path: Path) -> None:
    setup_router(tmp_path)
    client = TestClient(app)
    bt_id = _persist_backtest(tmp_path)

    resp = client.post(
        "/api/governance/models/from-backtest",
        json={"name": "gov-bt", "backtest_id": bt_id, "initial_stage": "candidate"},
    )
    assert resp.status_code == 200
    model = resp.json()
    model_id = model["id"]

    promote_resp = client.post(f"/api/governance/models/{model_id}/promote", json={"new_stage": "qualified"})
    assert promote_resp.status_code == 200

    list_resp = client.get("/api/governance/models")
    assert list_resp.status_code == 200
    assert any(item["id"] == model_id for item in list_resp.json())

    detail_resp = client.get(f"/api/governance/models/{model_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["kpi"]["total_return"] == 0.4
