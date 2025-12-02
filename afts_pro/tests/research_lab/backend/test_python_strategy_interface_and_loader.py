import sys
from pathlib import Path

import pytest

from research_lab.backend.core.python_strategies.loader import import_strategy_class, extract_metadata
from research_lab.backend.core.python_strategies.interface import PythonStrategyInterface


STRATEGY_CODE = """
from research_lab.backend.core.python_strategies.interface import BasePythonStrategy

class MyTestStrategy(BasePythonStrategy):
    strategy_key = "test.my_strategy"
    strategy_name = "My Test Strategy"
    strategy_version = "1.0.0"
    strategy_description = "Test-only strategy."
    strategy_tags = ["test", "demo"]
    strategy_params_schema = {"risk_pct": {"type": "float", "default": 1.0}}

    def initialize(self, params=None):
        super().initialize(params)
"""


def test_import_and_extract_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "my_strategy_loader"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(STRATEGY_CODE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    # ensure clean module state if reused
    if module_name in sys.modules:
        sys.modules.pop(module_name)

    cls = import_strategy_class(module_name, "MyTestStrategy")
    assert issubclass(cls, PythonStrategyInterface)

    metadata = extract_metadata(cls)
    assert metadata.key == "test.my_strategy"
    assert metadata.name == "My Test Strategy"
    assert metadata.version == "1.0.0"
    assert "risk_pct" in metadata.params_schema
