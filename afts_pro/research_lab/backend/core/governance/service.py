"""Governance service for registering, promoting, and scoring models."""

from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import List, Optional

from research_lab.backend.core.backtests.persistence import BacktestPersistence
from research_lab.backend.core.experiments.persistence import ExperimentPersistence
from research_lab.backend.core.governance.models import (
    BacktestLink,
    FtmoSnapshot,
    GovernanceScore,
    GovernanceTag,
    KpiSnapshot,
    ModelEntry,
    ModelEntrySummary,
    ModelStage,
    ModelType,
    RlLink,
    RlSnapshot,
)
from research_lab.backend.core.governance.registry import GovernanceRegistry
from research_lab.backend.core.rl.service import RLService
from research_lab.backend.core.rl_experiments.persistence import RlExperimentPersistence
from research_lab.backend.core.utils.datetime import ensure_utc_datetime


class GovernanceService:
    """Application service for the governance model hub."""

    def __init__(
        self,
        registry: GovernanceRegistry,
        backtest_persistence: BacktestPersistence,
        experiment_persistence: ExperimentPersistence,
        rl_service: RLService,
        rl_experiment_persistence: RlExperimentPersistence,
    ) -> None:
        self.registry = registry
        self.backtest_persistence = backtest_persistence
        self.experiment_persistence = experiment_persistence
        self.rl_service = rl_service
        self.rl_experiment_persistence = rl_experiment_persistence

    def register_from_backtest(
        self,
        name: str,
        backtest_id: str,
        experiment_id: str | None = None,
        experiment_run_id: str | None = None,
        initial_stage: ModelStage = ModelStage.CANDIDATE,
        tags: List[GovernanceTag] | None = None,
    ) -> ModelEntry:
        """Register a model entry from a backtest result."""

        result = self.backtest_persistence.load_result(backtest_id)
        if result is None:
            raise ValueError("Backtest result not found")

        kpi = KpiSnapshot(
            total_return=result.kpi_summary.total_return,
            profit_factor=result.kpi_summary.profit_factor,
            max_drawdown_pct=result.kpi_summary.max_drawdown,
            trade_count=result.kpi_summary.trade_count,
        )
        ftmo = FtmoSnapshot()
        if result.ftmo_risk_summary:
            ftmo_summary = result.ftmo_risk_summary
            ftmo = FtmoSnapshot(
                passed=ftmo_summary.passed,
                first_breach_type=ftmo_summary.first_breach.breach_type if ftmo_summary.first_breach else None,
                worst_daily_drawdown_pct=ftmo_summary.worst_daily_drawdown_pct,
                worst_total_drawdown_pct=ftmo_summary.worst_total_drawdown_pct,
            )

        now = datetime.now(timezone.utc)
        entry = ModelEntry(
            id=str(uuid.uuid4()),
            name=name,
            type=ModelType.BACKTEST_STRATEGY,
            stage=initial_stage,
            created_at=now,
            updated_at=now,
            backtest_link=BacktestLink(
                backtest_id=backtest_id,
                experiment_id=experiment_id,
                experiment_run_id=experiment_run_id,
            ),
            kpi=kpi,
            ftmo=ftmo,
            rl=RlSnapshot(),
            score=GovernanceScore(),
            tags=tags or [],
            metadata={},
        )
        self.registry.upsert_model(entry)
        return entry

    def register_from_rl_run(
        self,
        name: str,
        rl_run_id: str,
        rl_experiment_id: str | None = None,
        rl_experiment_run_id: str | None = None,
        initial_stage: ModelStage = ModelStage.CANDIDATE,
        tags: List[GovernanceTag] | None = None,
    ) -> ModelEntry:
        """Register a model entry from an RL run (job id)."""

        status, result, _ = self.rl_service.get_job_result(rl_run_id)
        if result is None:
            raise ValueError("RL run result not found")

        metrics = result.metrics
        rl_snapshot = RlSnapshot(
            mean_return=metrics.avg_reward,
            std_return=None,
            max_return=metrics.max_reward,
            steps=metrics.reward_curve[-1].step if metrics.reward_curve else None,
            reward_checks_passed=result.reward_check_result.passed if result.reward_check_result else None,
            failed_checks=[],
        )
        if result.reward_check_result and not result.reward_check_result.passed and result.reward_check_result.reason:
            rl_snapshot.failed_checks.append(result.reward_check_result.reason)

        now = datetime.now(timezone.utc)
        entry = ModelEntry(
            id=str(uuid.uuid4()),
            name=name,
            type=ModelType.RL_POLICY,
            stage=initial_stage,
            created_at=now,
            updated_at=now,
            backtest_link=None,
            rl_link=RlLink(
                rl_run_id=rl_run_id,
                rl_experiment_id=rl_experiment_id,
                rl_experiment_run_id=rl_experiment_run_id,
            ),
            kpi=KpiSnapshot(),
            ftmo=FtmoSnapshot(),
            rl=rl_snapshot,
            score=GovernanceScore(),
            tags=tags or [],
            metadata={},
        )
        self.registry.upsert_model(entry)
        return entry

    def promote(self, model_id: str, new_stage: ModelStage, note: str | None = None) -> ModelEntry:
        """Promote or demote a model to a new stage."""

        entry = self.registry.get_model(model_id)
        if entry is None:
            raise ValueError("Model not found")
        entry.stage = new_stage
        entry.updated_at = datetime.now(timezone.utc)
        if note:
            entry.metadata["promotion_note"] = note
        self.registry.upsert_model(entry)
        return entry

    def update_score(self, model_id: str, composite_score: float, note: str | None = None) -> ModelEntry:
        """Update governance score for a model."""

        entry = self.registry.get_model(model_id)
        if entry is None:
            raise ValueError("Model not found")
        entry.score.composite_score = composite_score
        entry.score.notes = note
        entry.updated_at = datetime.now(timezone.utc)
        self.registry.upsert_model(entry)
        return entry

    def list_models(self, stage: ModelStage | None = None, type: ModelType | None = None) -> List[ModelEntrySummary]:
        """List models with optional filters."""

        summaries = self.registry.list_models()
        if stage:
            summaries = [s for s in summaries if s.stage == stage]
        if type:
            summaries = [s for s in summaries if s.type == type]
        return summaries

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Fetch a model entry by id."""

        return self.registry.get_model(model_id)


__all__ = ["GovernanceService"]
