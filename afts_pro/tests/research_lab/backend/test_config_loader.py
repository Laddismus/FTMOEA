from pathlib import Path

import pytest
import yaml

from research_lab.backend.core.config_loader import ResearchConfigLoader


def test_config_loader_loads_yaml_and_lists(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "research"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "test_config.yaml"
    expected = {"foo": "bar", "nested": {"value": 1}}
    config_path.write_text(yaml.safe_dump(expected), encoding="utf-8")

    loader = ResearchConfigLoader(base_path=config_dir)

    loaded = loader.load_config("test_config")
    assert loaded == expected

    available = loader.list_configs()
    assert "test_config" in available


def test_config_loader_missing_file(tmp_path: Path) -> None:
    loader = ResearchConfigLoader(base_path=tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load_config("does_not_exist")


def test_config_loader_invalid_yaml(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "research"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "broken.yaml"
    config_path.write_text("foo: [unterminated", encoding="utf-8")

    loader = ResearchConfigLoader(base_path=config_dir)

    with pytest.raises(ValueError):
        loader.load_config("broken")
