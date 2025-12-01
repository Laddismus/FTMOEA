from pathlib import Path

from afts_pro.core.train_controller import TrainController, TrainJobConfig
from afts_pro.rl.rl_inference import RLInferenceHook, ObservationBuilder
from afts_pro.rl.types import RLObsSpec
from afts_pro.core.models import StrategyDecision, MarketState
from afts_pro.exec.position_models import AccountState
import pandas as pd


def test_trained_policy_can_be_loaded_for_sim_inference(tmp_path):
    env_cfg = tmp_path / "env.yaml"
    env_cfg.write_text("env_type: risk\nobservation: {}\n")
    agent_cfg = tmp_path / "agent.yaml"
    agent_cfg.write_text("action_mode: continuous\n")
    output_dir = tmp_path / "out"
    job = TrainJobConfig(
        agent_type="risk",
        env_config_path=str(env_cfg),
        agent_config_path=str(agent_cfg),
        output_dir=str(output_dir),
    )
    controller = TrainController()
    controller.run_train_job(job)

    obs_spec = RLObsSpec(shape=(4,), dtype="float32", as_dict=False)
    obs_builder = ObservationBuilder(obs_spec)
    # Load via RLInferenceHook using checkpoint dir
    from afts_pro.core.rl_hook_integration import integrate_rl_inference

    hook = integrate_rl_inference(
        use_risk_agent=True,
        use_exit_agent=False,
        risk_agent_path=str(output_dir),
        exit_agent_path=None,
        obs_spec=obs_spec,
    )
    assert hook is not None
    ms = MarketState(timestamp=pd.Timestamp.utcnow(), symbol="ETH", open=0, high=0, low=0, close=0, volume=0)
    acc = AccountState(balance=100, equity=100, realized_pnl=0, unrealized_pnl=0, fees_total=0)
    decision = StrategyDecision(action="entry", side="long", confidence=1.0)
    actions = hook.compute_actions(ms, acc, None, None)
    hook.apply_to_decision(decision, actions)
    assert decision.update.get("risk_pct") is not None
