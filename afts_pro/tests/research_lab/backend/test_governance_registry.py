from datetime import datetime, timezone
from pathlib import Path

from research_lab.backend.core.governance.models import ModelEntry, ModelStage, ModelType
from research_lab.backend.core.governance.registry import GovernanceRegistry


def _entry(entry_id: str) -> ModelEntry:
    now = datetime.now(timezone.utc)
    return ModelEntry(
        id=entry_id,
        name="model",
        type=ModelType.BACKTEST_STRATEGY,
        stage=ModelStage.CANDIDATE,
        created_at=now,
        updated_at=now,
    )


def test_registry_upsert_get_list_delete(tmp_path: Path) -> None:
    registry = GovernanceRegistry(tmp_path)
    entry = _entry("m1")

    registry.upsert_model(entry)
    fetched = registry.get_model("m1")
    assert fetched is not None
    assert fetched.id == "m1"

    summaries = registry.list_models()
    assert len(summaries) == 1
    assert summaries[0].id == "m1"

    registry.delete_model("m1")
    assert registry.get_model("m1") is None
