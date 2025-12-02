"""Persistence utilities for backtest results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from research_lab.backend.core.backtests.models import BacktestIndexEntry, BacktestResult
from research_lab.backend.core.utils.datetime import ensure_utc_datetime


class BacktestPersistence:
    """Persist backtest results as JSON artifacts and list/load them."""

    def __init__(self, backtests_dir: Path) -> None:
        self.backtests_dir = backtests_dir
        self.backtests_dir.mkdir(parents=True, exist_ok=True)

    def save_result(self, result: BacktestResult) -> Path:
        """Persist a backtest result to disk."""

        target = self.backtests_dir / f"{result.id}.json"
        payload = result.model_dump(mode="json")
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return target

    def load_result(self, run_id: str) -> Optional[BacktestResult]:
        """Load a backtest result by ID."""

        path = self.backtests_dir / f"{run_id}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return BacktestResult(**data)

    def list_runs(self) -> List[BacktestIndexEntry]:
        """List available backtest runs using stored artifacts."""

        entries: List[BacktestIndexEntry] = []
        for file in sorted(self.backtests_dir.glob("*.json")):
            with file.open("r", encoding="utf-8") as handle:
                try:
                    data = json.load(handle)
                except json.JSONDecodeError:
                    continue
            created_at = data.get("created_at")
            if created_at is None:
                created_at = datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc).isoformat()
            entries.append(
                BacktestIndexEntry(
                    id=data.get("id", file.stem),
                    created_at=ensure_utc_datetime(created_at),
                    mode=data.get("mode"),
                    metadata=data.get("metadata", {}),
                    kpi_summary=data.get("kpi_summary"),
                )
            )
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries


__all__ = ["BacktestPersistence"]
