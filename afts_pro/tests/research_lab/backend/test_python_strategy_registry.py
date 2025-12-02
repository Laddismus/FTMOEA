from pathlib import Path

from research_lab.backend.core.python_strategies.models import PythonStrategyMetadata
from research_lab.backend.core.python_strategies.registry import PythonStrategyRegistry


def test_registry_persists_and_updates(tmp_path: Path) -> None:
    registry = PythonStrategyRegistry(registry_dir=tmp_path)

    meta1 = PythonStrategyMetadata(
        key="strategy.one",
        name="One",
        version="1.0.0",
        module_path="module.one",
        class_name="ClassOne",
        description=None,
        tags=["demo"],
        params_schema={"a": 1},
    )
    registry.register_strategy(meta1)

    loaded = registry.get_strategy("strategy.one")
    assert loaded is not None
    assert loaded.name == "One"

    meta_updated = meta1.model_copy(update={"version": "1.1.0"})
    registry.register_strategy(meta_updated)
    loaded_updated = registry.get_strategy("strategy.one")
    assert loaded_updated.version == "1.1.0"

    all_strategies = registry.list_strategies()
    assert any(s.key == "strategy.one" for s in all_strategies)
