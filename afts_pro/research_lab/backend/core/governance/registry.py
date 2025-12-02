"""File-based registry for governance model entries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from research_lab.backend.core.governance.models import ModelEntry, ModelEntrySummary


class GovernanceRegistry:
    """Persist governance model entries to a JSON index."""

    def __init__(self, governance_dir: Path) -> None:
        self.governance_dir = governance_dir
        self.index_path = governance_dir / "models_index.json"
        self.governance_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> List[ModelEntry]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return [ModelEntry(**item) for item in data]
        except json.JSONDecodeError:
            return []

    def _write_index(self, entries: List[ModelEntry]) -> None:
        payload = [entry.model_dump(mode="json") for entry in entries]
        self.index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list_models(self) -> List[ModelEntrySummary]:
        """Return summaries of all models in the registry."""

        return [self._to_summary(entry) for entry in self._load_index()]

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Return a model entry by id."""

        for entry in self._load_index():
            if entry.id == model_id:
                return entry
        return None

    def upsert_model(self, entry: ModelEntry) -> None:
        """Insert or update a model entry."""

        entries = self._load_index()
        updated = False
        for idx, item in enumerate(entries):
            if item.id == entry.id:
                entries[idx] = entry
                updated = True
                break
        if not updated:
            entries.append(entry)
        self._write_index(entries)

    def delete_model(self, model_id: str) -> None:
        """Delete a model entry from the registry."""

        entries = [e for e in self._load_index() if e.id != model_id]
        self._write_index(entries)

    def _to_summary(self, entry: ModelEntry) -> ModelEntrySummary:
        return ModelEntrySummary(
            id=entry.id,
            name=entry.name,
            type=entry.type,
            stage=entry.stage,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
            total_return=entry.kpi.total_return,
            profit_factor=entry.kpi.profit_factor,
            mean_return=entry.rl.mean_return,
            ftmo_passed=entry.ftmo.passed,
        )


__all__ = ["GovernanceRegistry"]
