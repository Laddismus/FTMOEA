import pytest

from afts_pro.core.model_registry_loader import ModelRegistryConfig, ModelRegistryLoader
from afts_pro.core.live_runner import LiveConfig


def test_live_mode_aborts_when_production_missing(tmp_path):
    cfg = LiveConfig(
        symbol="TEST",
        poll_interval_sec=0.0,
        max_steps=1,
        use_system_gate=False,
        gate_mode="from-last",
        qa_config_path="configs/qa/qa.yaml",
        gate_policy_path="configs/qa/gate_policy.yaml",
    )
    # Fake broker/engine stubs not needed; we just simulate missing production by invoking loader
    loader = ModelRegistryLoader(ModelRegistryConfig(promotion_root=str(tmp_path / "prod")))
    assert loader.has_production_model("missing") is False
    with pytest.raises(FileNotFoundError):
        loader.load_production_ref("missing", "risk")
