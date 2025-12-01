from pathlib import Path

from afts_pro.config.profile_config import load_profile
from afts_pro.config.global_config import load_global_config_from_profile
from afts_pro.config.loader import load_yaml


def test_env_config_loads_and_has_reward_profiles():
    data = load_yaml("configs/rl/env.yaml")
    assert "reward_profiles" in data


def test_train_profiles_resolve_paths(tmp_path):
    profile = load_profile("configs/profiles/sim.yaml")
    assert profile.includes.environment


def test_sim_mode_config_parses_rl_flags(tmp_path):
    cfg_path = tmp_path / "sim.yaml"
    cfg_path.write_text("use_risk_agent: true\nuse_exit_agent: true\n")
    data = load_yaml(str(cfg_path))
    assert data.get("use_risk_agent") is True
