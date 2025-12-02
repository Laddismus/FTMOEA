from pathlib import Path

from research_lab.backend.core.backtests.engine import RollingKpiBacktestEngine
from research_lab.backend.core.backtests.models import BacktestRequest
from research_lab.backend.core.backtests.persistence import BacktestPersistence
from research_lab.backend.core.backtests.service import BacktestService
from research_lab.backend.core.job_runner import InMemoryJobRunner


def build_service(tmp_path: Path) -> BacktestService:
    return BacktestService(
        job_runner=InMemoryJobRunner(),
        engine=RollingKpiBacktestEngine(),
        persistence=BacktestPersistence(tmp_path),
    )


def test_run_sync_returns_result(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    request = BacktestRequest(mode="graph", returns=[1, -0.5, 0.5], window=2)

    result = service.run_sync(request)

    assert result.kpi_summary.total_return == sum(request.returns)


def test_submit_job_and_get_result(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    request = BacktestRequest(mode="graph", returns=[1, -1, 1], window=2)

    job_id = service.submit_job(request)
    status = service.get_job_result(job_id)

    assert status is not None
    assert status["status"] == "completed"
    assert status["result"]["kpi_summary"]["total_return"] == sum(request.returns)
