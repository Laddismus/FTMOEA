from pathlib import Path
import types

from afts_pro.core.train_controller import TrainController, TrainJobConfig


class DummySummary:
    def __init__(self):
        self.episodes = 1
        self.mean_return = 0.0
        self.best_return = 0.0


def test_train_controller_builds_risk_job_and_calls_loop(monkeypatch, tmp_path):
    env_cfg = tmp_path / "env.yaml"
    env_cfg.write_text("env_type: risk\nobservation: {}\n")
    agent_cfg = tmp_path / "agent.yaml"
    agent_cfg.write_text("action_mode: continuous\n")
    called = {}

    def fake_train(*args, **kwargs):
        called["risk"] = True
        return DummySummary()

    monkeypatch.setattr("afts_pro.core.train_controller.train_risk_agent", fake_train)
    controller = TrainController()
    job = TrainJobConfig(agent_type="risk", env_config_path=str(env_cfg), agent_config_path=str(agent_cfg), output_dir=str(tmp_path / "out"))
    result = controller.run_train_job(job)
    assert called.get("risk") is True
    assert Path(result.output_dir).exists()


def test_train_controller_builds_exit_job_and_calls_loop(monkeypatch, tmp_path):
    env_cfg = tmp_path / "env.yaml"
    env_cfg.write_text("env_type: exit\nobservation: {}\n")
    agent_cfg = tmp_path / "agent.yaml"
    agent_cfg.write_text("action_mode: discrete\nn_actions: 6\n")
    called = {}

    def fake_train(*args, **kwargs):
        called["exit"] = True
        return DummySummary()

    monkeypatch.setattr("afts_pro.core.train_controller.train_exit_agent", fake_train)
    controller = TrainController()
    job = TrainJobConfig(agent_type="exit", env_config_path=str(env_cfg), agent_config_path=str(agent_cfg), output_dir=str(tmp_path / "out2"))
    result = controller.run_train_job(job)
    assert called.get("exit") is True
    assert Path(result.output_dir).exists()
