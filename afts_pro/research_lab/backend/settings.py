"""Central settings for the Research Lab backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass
class ResearchSettings:
    """Holds filesystem locations for the research backend."""

    project_root: Path = Path(__file__).resolve().parents[2]
    config_root: Path = project_root / "configs" / "research"
    artifacts_root: Path = project_root / "artifacts" / "research"
    strategies_dir: Path = config_root / "strategies"
    python_strategies_dir: Path = project_root / "strategies" / "custom"
    backtests_dir: Path = artifacts_root / "backtests"
    default_timezone: str = "UTC"

    @property
    def model_registry_root(self) -> Path:
        return self.artifacts_root / "models"

    @property
    def model_registry_index(self) -> Path:
        return self.model_registry_root / "registry.json"

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.default_timezone)

    @property
    def experiments_dir(self) -> Path:
        return self.artifacts_root / "experiments"

    @property
    def rl_policies_dir(self) -> Path:
        return self.artifacts_root / "rl_policies"

    @property
    def rl_runs_dir(self) -> Path:
        return self.artifacts_root / "rl_runs"

    @property
    def rl_experiments_dir(self) -> Path:
        return self.artifacts_root / "rl_experiments"

    @property
    def governance_dir(self) -> Path:
        return self.artifacts_root / "governance"


def get_settings() -> ResearchSettings:
    """Return research backend settings."""

    return ResearchSettings()


__all__ = ["ResearchSettings", "get_settings"]
