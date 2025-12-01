import yaml
from pathlib import Path

from afts_pro.core.train_controller import TrainController, TrainJobConfig


def test_train_profile_configs_load():
    data = yaml.safe_load(Path("configs/train_profiles.yaml").read_text())
    profiles = data.get("profiles", {})
    assert "orb_eurusd_riskagent_v1" in profiles
    assert "orb_eurusd_exitagent_v1" in profiles
    for key in ("env_config", "agent_config"):
        assert Path(profiles["orb_eurusd_riskagent_v1"][key]).exists()
        assert Path(profiles["orb_eurusd_exitagent_v1"][key]).exists()


def test_train_controller_builds_job_from_orb_profile(tmp_path):
    data = yaml.safe_load(Path("configs/train_profiles.yaml").read_text())
    profile = data["profiles"]["orb_eurusd_riskagent_v1"]
    output_dir = tmp_path / "risk_job"
    job_cfg = TrainJobConfig(
        agent_type=profile["agent_type"],
        env_config_path=profile["env_config"],
        agent_config_path=profile["agent_config"],
        output_dir=str(output_dir),
    )
    controller = TrainController()
    result = controller.run_train_job(job_cfg)
    assert Path(result.output_dir).exists()
    assert Path(result.output_dir, "train_job_config.yaml").exists()


def test_train_smoke_run_exit_agent(tmp_path):
    data = yaml.safe_load(Path("configs/train_profiles.yaml").read_text())
    profile = data["profiles"]["orb_eurusd_exitagent_v1"]
    output_dir = tmp_path / "exit_job"
    job_cfg = TrainJobConfig(
        agent_type=profile["agent_type"],
        env_config_path=profile["env_config"],
        agent_config_path=profile["agent_config"],
        output_dir=str(output_dir),
    )
    controller = TrainController()
    result = controller.run_train_job(job_cfg)
    assert Path(result.output_dir).exists()
    assert Path(result.output_dir, "train_job_config.yaml").exists()
