import pytest

from afts_pro.core.model_registry_loader import ModelRegistryConfig, ModelRegistryLoader
from afts_pro.core.rl_hook_integration import integrate_rl_inference
from afts_pro.rl.types import RLObsSpec


def test_sim_uses_production_models_when_enabled(tmp_path):
    # prepare production refs
    prod_root = tmp_path / "models" / "production"
    risk_tag_dir = prod_root / "orb_risk"
    exit_tag_dir = prod_root / "orb_exit"
    risk_tag_dir.mkdir(parents=True)
    exit_tag_dir.mkdir(parents=True)
    risk_ckpt = risk_tag_dir / "risk.pt"
    exit_ckpt = exit_tag_dir / "exit.pt"
    risk_ckpt.write_text("dummy")
    exit_ckpt.write_text("dummy")
    (risk_tag_dir / "CURRENT.txt").write_text(str(risk_ckpt))
    (exit_tag_dir / "CURRENT.txt").write_text(str(exit_ckpt))

    profile = {
        "promotion_root": str(prod_root),
        "risk_tag": "orb_risk",
        "exit_tag": "orb_exit",
    }
    hook = integrate_rl_inference(
        use_risk_agent=True,
        use_exit_agent=True,
        risk_agent_path=None,
        exit_agent_path=None,
        obs_spec=RLObsSpec(shape=(4,)),
        use_production_models=True,
        production_profile=profile,
    )
    assert hook is not None


def test_sim_prod_fails_when_no_production_model(tmp_path):
    prod_root = tmp_path / "models" / "production"
    prod_root.mkdir(parents=True)
    profile = {"promotion_root": str(prod_root), "risk_tag": "none", "exit_tag": "none"}
    with pytest.raises(FileNotFoundError):
        integrate_rl_inference(
            use_risk_agent=True,
            use_exit_agent=False,
            risk_agent_path=None,
            exit_agent_path=None,
            obs_spec=RLObsSpec(shape=(4,)),
            use_production_models=True,
            production_profile=profile,
        )
