import yaml
from pathlib import Path

from afts_pro.core.train_controller import TrainController, TrainJobConfig
from afts_pro.core import Mode, ModeDispatcher


def test_train_controller_runs_risk_job(tmp_path):
    env_cfg = tmp_path / "env.yaml"
    env_cfg.write_text("env_type: risk\nobservation: {}\n")
    agent_cfg = tmp_path / "agent.yaml"
    agent_cfg.write_text("action_mode: continuous\nn_actions: 1\n")
    output_dir = tmp_path / "out"
    job = TrainJobConfig(
        agent_type="risk",
        env_config_path=str(env_cfg),
        agent_config_path=str(agent_cfg),
        output_dir=str(output_dir),
        seed=1,
    )
    controller = TrainController()
    result = controller.run_train_job(job)
    assert Path(result.output_dir).exists()
    assert result.agent_type == "risk"


def test_train_controller_runs_exit_job(tmp_path):
    env_cfg = tmp_path / "env.yaml"
    env_cfg.write_text("env_type: exit\nobservation: {}\n")
    agent_cfg = tmp_path / "agent.yaml"
    agent_cfg.write_text("action_mode: discrete\nn_actions: 6\n")
    output_dir = tmp_path / "out2"
    job = TrainJobConfig(
        agent_type="exit",
        env_config_path=str(env_cfg),
        agent_config_path=str(agent_cfg),
        output_dir=str(output_dir),
        seed=1,
    )
    controller = TrainController()
    result = controller.run_train_job(job)
    assert Path(result.output_dir).exists()
    assert result.agent_type == "exit"
