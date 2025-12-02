"""Persistence layer for research experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from research_lab.backend.core.experiments.models import (
    ExperimentConfig,
    ExperimentRunStatus,
    ExperimentStatus,
    ExperimentSummary,
)
from research_lab.backend.core.utils.datetime import ensure_utc_datetime


class ExperimentPersistence:
    """Persist experiment configurations and run statuses as JSON artifacts."""

    def __init__(self, experiments_dir: Path) -> None:
        self.experiments_dir = experiments_dir
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def save_experiment(self, config: ExperimentConfig, runs: List[ExperimentRunStatus]) -> None:
        """Persist an experiment configuration with its run statuses."""

        payload = {"config": config.model_dump(mode="json"), "runs": [run.model_dump(mode="json") for run in runs]}
        target = self.experiments_dir / f"{config.id}.json"
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load_experiment(self, experiment_id: str) -> Optional[Tuple[ExperimentConfig, List[ExperimentRunStatus]]]:
        """Load an experiment by its identifier."""

        path = self.experiments_dir / f"{experiment_id}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        config = ExperimentConfig(**data["config"])
        runs = [ExperimentRunStatus(**run) for run in data.get("runs", [])]
        return config, runs

    def list_experiments(self) -> List[ExperimentSummary]:
        """List experiment summaries derived from stored artifacts."""

        summaries: List[ExperimentSummary] = []
        for file in sorted(self.experiments_dir.glob("*.json")):
            with file.open("r", encoding="utf-8") as handle:
                try:
                    data = json.load(handle)
                except json.JSONDecodeError:
                    continue
            cfg_data = data.get("config")
            runs_data = data.get("runs", [])
            if not cfg_data:
                continue
            config = ExperimentConfig(**cfg_data)
            runs = [ExperimentRunStatus(**run) for run in runs_data]
            total_runs = len(runs)
            completed_runs = len([r for r in runs if r.status == ExperimentStatus.COMPLETED])
            failed_runs = len([r for r in runs if r.status == ExperimentStatus.FAILED])
            status = ExperimentStatus.PENDING
            if failed_runs > 0:
                status = ExperimentStatus.FAILED
            elif any(r.status == ExperimentStatus.RUNNING for r in runs):
                status = ExperimentStatus.RUNNING
            elif total_runs > 0 and completed_runs == total_runs:
                status = ExperimentStatus.COMPLETED
            summary = ExperimentSummary(
                id=config.id,
                name=config.name,
                status=status,
                created_at=ensure_utc_datetime(config.created_at),
                total_runs=total_runs,
                completed_runs=completed_runs,
                failed_runs=failed_runs,
                best_total_return=None,
                best_pf=None,
            )
            summaries.append(summary)
        summaries.sort(key=lambda s: s.created_at, reverse=True)
        return summaries


__all__ = ["ExperimentPersistence"]
