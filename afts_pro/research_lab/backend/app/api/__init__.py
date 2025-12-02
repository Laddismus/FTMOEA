"""API router collection for the Research Lab backend."""

from fastapi import APIRouter

from research_lab.backend.app.api import (
    health,
    jobs,
    registry,
    analytics,
    strategy_builder,
    python_strategies,
    backtests,
    experiments,
    rl,
    rl_experiments,
    governance,
    debug,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(registry.router, prefix="/registry")
api_router.include_router(jobs.router, prefix="/jobs")
api_router.include_router(analytics.router)
api_router.include_router(strategy_builder.router)
api_router.include_router(python_strategies.router)
api_router.include_router(backtests.router)
api_router.include_router(experiments.router)
api_router.include_router(rl.router)
api_router.include_router(rl_experiments.router)
api_router.include_router(governance.router)
api_router.include_router(debug.router)

__all__ = ["api_router"]
