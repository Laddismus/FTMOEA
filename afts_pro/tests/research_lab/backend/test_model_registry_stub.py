from pathlib import Path

from research_lab.backend.core.model_registry import FileSystemModelRegistry


def test_model_registry_stub(tmp_path: Path) -> None:
    registry_root = tmp_path / "artifacts" / "research" / "models"
    registry = FileSystemModelRegistry(registry_root=registry_root)
    model_path = tmp_path / "models" / "test_model.bin"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("content", encoding="utf-8")

    registry.register_model("test_model", "v1", model_path, {"foo": "bar"})

    resolved = registry.get_model("test_model", "v1")
    assert resolved == model_path

    models = registry.list_models()
    assert any(entry["name"] == "test_model" and entry["version"] == "v1" for entry in models)
