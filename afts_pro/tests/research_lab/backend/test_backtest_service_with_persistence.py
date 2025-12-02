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


def test_run_sync_persists_result(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    request = BacktestRequest(mode="graph", returns=[1, -0.5, 0.5], window=2)

    result = service.run_sync(request)

    assert result.id
    assert (tmp_path / f"{result.id}.json").exists()


def test_submit_job_and_get_run(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    request = BacktestRequest(mode="graph", returns=[1, -1, 1], window=2)

    job_id = service.submit_job(request)
    status = service.get_job_result(job_id)
    assert status is not None
    run_id = status["result"]["id"]

    loaded = service.get_run(run_id)
    assert loaded is not None
    assert loaded.id == run_id
    assert loaded.kpi_summary.total_return == sum(request.returns)
