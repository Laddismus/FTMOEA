from pathlib import Path

import pytest

from afts_pro.core.model_registry_loader import ModelRegistryConfig, ModelRegistryLoader


def test_load_production_ref_reads_current_and_meta(tmp_path):
    tag_dir = tmp_path / "orb_risk"
    tag_dir.mkdir()
    ckpt = tag_dir / "ckpt_risk.pt"
    ckpt.write_text("dummy")
    (tag_dir / "CURRENT.txt").write_text(str(ckpt))
    meta = {"score": 1.0}
    (tag_dir / "selection_info.json").write_text('{"score":1.0}')
    loader = ModelRegistryLoader(ModelRegistryConfig(promotion_root=str(tmp_path)))
    ref = loader.load_production_ref("orb_risk", "risk")
    assert ref.checkpoint_path == ckpt
    assert ref.selection_meta.get("score") == meta["score"]


def test_has_production_model_true_false(tmp_path):
    loader = ModelRegistryLoader(ModelRegistryConfig(promotion_root=str(tmp_path)))
    tag_dir = tmp_path / "tag"
    tag_dir.mkdir()
    assert loader.has_production_model("tag") is False
    (tag_dir / "CURRENT.txt").write_text("missing")
    assert loader.has_production_model("tag") is True


def test_load_production_ref_missing_files_raises(tmp_path):
    loader = ModelRegistryLoader(ModelRegistryConfig(promotion_root=str(tmp_path)))
    with pytest.raises(FileNotFoundError):
        loader.load_production_ref("missing", "risk")
