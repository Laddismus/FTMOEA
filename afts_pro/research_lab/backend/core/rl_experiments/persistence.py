"""Persistence utilities for RL experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from research_lab.backend.core.rl_experiments.models import (
    RlExperimentConfig,
    RlExperimentRunStatus,
)


class RlExperimentPersistence:
    """Persist RL experiment configs and runs as JSON artifacts."""

    def __init__(self, experiments_dir: Path) -> None:
        self.experiments_dir = experiments_dir
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def save_experiment(self, config: RlExperimentConfig, runs: List[RlExperimentRunStatus]) -> None:
        payload = {"config": config.model_dump(mode="json"), "runs": [run.model_dump(mode="json") for run in runs]}
        target = self.experiments_dir / f"{config.id}.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_experiment(self, experiment_id: str) -> Optional[Tuple[RlExperimentConfig, List[RlExperimentRunStatus]]]:
        path = self.experiments_dir / f"{experiment_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        config = RlExperimentConfig(**data["config"])
        runs = [RlExperimentRunStatus(**run) for run in data.get("runs", [])]
        return config, runs

    def list_experiments(self) -> List[RlExperimentConfig]:
        configs: List[RlExperimentConfig] = []
        for file in sorted(self.experiments_dir.glob("*.json")):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                configs.append(RlExperimentConfig(**data["config"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return configs


__all__ = ["RlExperimentPersistence"]
